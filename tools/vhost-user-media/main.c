/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Vhost-user-media main entry point.
 * Uses libvhost-user-glib to interface with QEMU.
 *
 * Copyright 2025-2026 NXP
 */

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <linux/videodev2.h>
#include <glib.h>
#include <glib-unix.h>
#include <gio/gio.h>
#include <gio/gunixsocketaddress.h>

#include <xengnttab.h>

#include "libvhost-user-glib.h"
#include "libvhost-user.h"

#include "vhost-user-media.h"

#define VHOST_USER_MEDIA_MAX_QUEUES 2

#ifndef container_of
#define container_of(ptr, type, member) \
    ((type *)((char *)(ptr) - offsetof(type, member)))
#endif

struct vu_media_ctx;

struct vu_media_device {
    VugDev dev;
    struct vhost_user_media media;

    /* Back pointer to the process-wide shared context. */
    struct vu_media_ctx *ctx;

    /* Event queue state */
    VuVirtq *eventq;
    GList *event_bufs; /* Available event buffer descriptors */

    /*
     * Events produced while the guest's event virtqueue had no available
     * buffer. They must not be dropped (e.g. a DQBUF event carrying
     * V4L2_BUF_FLAG_LAST): losing one wedges the guest pipeline waiting for
     * EOS. Queued here in FIFO order and flushed once the guest replenishes
     * the eventq.
     */
    GQueue *pending_events;

    /*
     * Persistent command/response scratch buffers, allocated once at startup
     * and reused for every command. Sized for the worst case (an MMAP response
     * carrying VIRTIO_MEDIA_MAX_MAPPING_PAGES grant refs) so 4K buffers no
     * longer overflow, while avoiding a malloc/free per command. One pair per
     * device: the single-threaded GLib main loop serialises commands, but
     * keeping them per-device avoids any cross-device coupling.
     */
    uint8_t *cmd_buf;
    size_t cmd_buf_size;
    uint8_t *resp_buf;
    size_t resp_buf_size;
};

/*
 * Process-wide shared context. One vhost-user-media process now drives several
 * virtio-media devices (one VuDev / socket / guest PCI device each) off a single
 * GLib main loop, so the resources that need to exist only once - the main
 * loop, the Xen grant device handle - live here and are shared by every device.
 * This is what lets foreign-map accounting be coordinated across devices (the
 * lever for capping Dom0 peak memory); see notes/single-daemon-refactor.md.
 */
struct vu_media_ctx {
    GMainLoop *loop;

    /*
     * Xen grant device for dma-buf import/export, opened once and shared by
     * every device's struct vhost_user_media (each media->gntdev_handle points
     * here). Grant refs are domain-global, so a single handle serves all.
     */
    void *gntdev_handle;

    GList *devices; /* struct vu_media_device * for every device */
};

/*
 * Command buffer size. Commands (OPEN/IOCTL/MMAP/...) are small; 8 KiB matches
 * the historical bound and comfortably holds the largest ioctl payload.
 */
#define VU_MEDIA_CMD_BUF_SIZE 8192

/*
 * Response buffer size. The MMAP response is the largest: a fixed header plus
 * up to VIRTIO_MEDIA_MAX_MAPPING_PAGES 32-bit grant refs.
 */
#define VU_MEDIA_RESP_BUF_SIZE \
    (sizeof(struct virtio_media_resp_mmap) + \
     VIRTIO_MEDIA_MAX_MAPPING_PAGES * sizeof(uint32_t))

/* A single event awaiting an available eventq buffer. */
struct pending_event {
    size_t len;
    uint8_t data[];
};

static struct vu_media_ctx vu_ctx;

/*
 * Recover the owning device from the libvhost-user VuDev handle passed to the
 * VuDevIface callbacks. VugDev.parent is the embedded VuDev, and VugDev is the
 * first member of struct vu_media_device, so a nested container_of walks back
 * to the device.
 */
static inline struct vu_media_device *vu_media_from_vudev(VuDev *vudev)
{
    VugDev *vugdev = container_of(vudev, VugDev, parent);
    return container_of(vugdev, struct vu_media_device, dev);
}

/*
 * Recover the owning device from the struct vhost_user_media passed to the
 * media-layer callbacks (send_event / register_poll).
 */
static inline struct vu_media_device *vu_media_from_media(
    struct vhost_user_media *m)
{
    return container_of(m, struct vu_media_device, media);
}

/*
 * Try to deliver one event into an available eventq buffer.
 * Returns true if delivered, false if the eventq had no available buffer.
 */
static bool vu_media_deliver_event(struct vu_media_device *dev,
                                   const void *event, size_t event_len)
{
    VuDev *vudev = &dev->dev.parent;
    VuVirtq *vq = vu_get_queue(vudev, 1); /* eventq is queue index 1 */
    VuVirtqElement *elem;
    size_t written = 0;
    int i;

    /* Pop an available descriptor from the event queue */
    elem = vu_queue_pop(vudev, vq, sizeof(VuVirtqElement));
    if (!elem)
        return false;

    /* Write the event into the device-writable descriptor */
    for (i = 0; i < elem->in_num && written < event_len; i++) {
        size_t chunk = event_len - written;
        if (chunk > elem->in_sg[i].iov_len)
            chunk = elem->in_sg[i].iov_len;
        memcpy(elem->in_sg[i].iov_base, (const uint8_t *)event + written,
               chunk);
        written += chunk;
    }

    /* Return the descriptor to the guest */
    vu_queue_push(vudev, vq, elem, written);
    vu_queue_notify(vudev, vq);
    g_free(elem);
    return true;
}

/*
 * Flush events that were queued while the eventq had no available buffer.
 * Delivers in FIFO order and stops as soon as the eventq runs dry again so
 * ordering is preserved.
 */
static void vu_media_flush_pending_events(struct vu_media_device *dev)
{
    while (!g_queue_is_empty(dev->pending_events)) {
        struct pending_event *pe = g_queue_peek_head(dev->pending_events);

        if (!vu_media_deliver_event(dev, pe->data, pe->len))
            break; /* still no buffer; keep the rest queued in order */

        g_queue_pop_head(dev->pending_events);
        g_free(pe);
    }
}

/*
 * Send an event to the guest via the event virtqueue.
 *
 * Events must never be dropped: losing a DQBUF event that carries
 * V4L2_BUF_FLAG_LAST wedges the guest pipeline waiting for EOS. If the eventq
 * has no available buffer (or events are already queued and must stay in
 * order), stash the event and flush it once the guest replenishes the eventq.
 */
static void vu_media_send_event(struct vhost_user_media *m,
                                const void *event, size_t event_len)
{
    struct vu_media_device *dev = vu_media_from_media(m);

    /* Preserve order: if events are already pending, this one goes behind. */
    if (g_queue_is_empty(dev->pending_events) &&
        vu_media_deliver_event(dev, event, event_len))
        return;

    struct pending_event *pe = g_malloc(sizeof(*pe) + event_len);
    pe->len = event_len;
    memcpy(pe->data, event, event_len);
    g_queue_push_tail(dev->pending_events, pe);

    /* The guest may already have buffers waiting; try to drain now. */
    vu_media_flush_pending_events(dev);
}

/*
 * Handle a command from the command virtqueue (queue index 0).
 */
static void vu_media_handle_commandq(VuDev *vudev, int qidx)
{
    struct vu_media_device *dev = vu_media_from_vudev(vudev);
    VuVirtq *vq = vu_get_queue(vudev, qidx);
    VuVirtqElement *elem;

    while ((elem = vu_queue_pop(vudev, vq, sizeof(VuVirtqElement)))) {
        /* Gather the device-readable (command) data */
        uint8_t *cmd_buf = dev->cmd_buf;
        size_t cmd_len = 0;
        int i;

        for (i = 0; i < elem->out_num && cmd_len < dev->cmd_buf_size; i++) {
            size_t chunk = elem->out_sg[i].iov_len;
            if (cmd_len + chunk > dev->cmd_buf_size)
                chunk = dev->cmd_buf_size - cmd_len;
            memcpy(cmd_buf + cmd_len, elem->out_sg[i].iov_base, chunk);
            cmd_len += chunk;
        }

        /* Prepare response buffer */
        uint8_t *resp_buf = dev->resp_buf;
        size_t resp_len = 0;
        size_t resp_buf_size = dev->resp_buf_size;

        /* Also calculate available device-writable space */
        size_t in_total = 0;
        for (i = 0; i < elem->in_num; i++)
            in_total += elem->in_sg[i].iov_len;
        if (resp_buf_size > in_total)
            resp_buf_size = in_total;

        /* Process the command */
        memset(resp_buf, 0, resp_buf_size);
        /*
         * Pick up the DomU domid negotiated over vhost-user. It is filled
         * in by libvhost-user-xen when the guest memory table is set, well
         * before any MMAP command, and reflects the real domain id even
         * across DomU recreations.
         */
        if (vudev->xen_domainid)
            dev->media.domu_domid = vudev->xen_domainid;
        /*
         * Let the command layer reach the VuDev for the shared-object APIs
         * (vu_add/rm_shared_object in handle_cmd_mmap / cleanup paths).
         */
        dev->media.opaque = vudev;
        vhost_media_process_command(&dev->media, cmd_buf, cmd_len,
                                    resp_buf, resp_buf_size, &resp_len);

        /* Write response back to device-writable descriptors */
        size_t written = 0;
        for (i = 0; i < elem->in_num && written < resp_len; i++) {
            size_t chunk = resp_len - written;
            if (chunk > elem->in_sg[i].iov_len)
                chunk = elem->in_sg[i].iov_len;
            memcpy(elem->in_sg[i].iov_base, resp_buf + written, chunk);
            written += chunk;
        }

        vu_queue_push(vudev, vq, elem, written);
        vu_queue_notify(vudev, vq);
        g_free(elem);
    }
}

/*
 * Handle the event queue (queue index 1).
 * The guest sends us empty buffers that we fill with events later. When it
 * replenishes the eventq, flush any events that were stashed because no buffer
 * was available at the time they were produced.
 */
static void vu_media_handle_eventq(VuDev *vudev, int qidx)
{
    vu_media_flush_pending_events(vu_media_from_vudev(vudev));
}

static void vu_media_queue_set_started(VuDev *vudev, int qidx, bool started)
{
    g_debug("Queue %d started=%d", qidx, started);

    if (qidx == 0) {
        vu_set_queue_handler(vudev, vu_get_queue(vudev, 0),
                             started ? vu_media_handle_commandq : NULL);
    } else if (qidx == 1) {
        vu_set_queue_handler(vudev, vu_get_queue(vudev, 1),
                             started ? vu_media_handle_eventq : NULL);
    }
}

static uint64_t vu_media_get_features(VuDev *vudev)
{
    return 0;
}

static void vu_media_set_features(VuDev *vudev, uint64_t features)
{
}

static int vu_media_get_config(VuDev *vudev, uint8_t *config, uint32_t len)
{
    struct vu_media_device *dev = vu_media_from_vudev(vudev);
    struct virtio_media_config cfg;
    struct v4l2_capability cap;
    int fd;

    memset(&cfg, 0, sizeof(cfg));

    /* Query the V4L2 device capabilities */
    fd = open(dev->media.v4l2_device, O_RDWR);
    if (fd >= 0) {
        if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
            cfg.device_caps = cap.device_caps;
            cfg.device_type = 0; /* VFL_TYPE_VIDEO */
            strncpy(cfg.card, (char *)cap.card, sizeof(cfg.card) - 1);
        }
        close(fd);
    }

    if (len > sizeof(cfg))
        len = sizeof(cfg);
    memcpy(config, &cfg, len);

    return 0;
}

static int vu_media_set_config(VuDev *vudev, const uint8_t *data,
                               uint32_t offset, uint32_t size,
                               uint32_t flags)
{
    /*
     * The virtio-media config space (device_caps/device_type/card) is
     * read-only and derived from the host V4L2 device. There is nothing
     * for the guest to set, so just accept the write.
     *
     * Note: libvhost-user only advertises VHOST_USER_PROTOCOL_F_CONFIG
     * when both get_config and set_config are provided, and the QEMU
     * frontend requires that feature when supports_config is enabled.
     */
    return 0;
}

static void vu_media_panic(VuDev *vudev, const char *msg)
{
    g_critical("vhost-user-media panic: %s", msg);
    if (vu_ctx.loop)
        g_main_loop_quit(vu_ctx.loop);
}

/*
 * Advertise VHOST_USER_PROTOCOL_F_SHARED_OBJECT on top of the libvhost-user
 * defaults. Required so QEMU forwards BACKEND_SHARED_OBJECT_ADD/REMOVE/LOOKUP
 * and GET_SHARED_OBJECT to/from this daemon (host-object model: host pages +
 * UUID). F_BACKEND_REQ, which carries these backend-channel messages, is
 * already in the libvhost-user default set.
 */
static uint64_t vu_media_get_protocol_features(VuDev *vudev)
{
    return 1ULL << VHOST_USER_PROTOCOL_F_SHARED_OBJECT;
}

/*
 * Search every device in this process for a live buffer carrying `uuid` and
 * return its owning device's dma_fd (NOT dup'd; the caller must not close it),
 * or -1 if none has it. Shared by the QEMU-facing producer callback and the
 * consumer-side local fast path.
 */
static int vu_ctx_lookup_shared_object(const uint8_t *uuid)
{
    GList *l;

    for (l = vu_ctx.devices; l; l = l->next) {
        struct vu_media_device *dev = l->data;
        int fd = vhost_media_lookup_shared_object(&dev->media, uuid);

        if (fd >= 0)
            return fd;
    }
    return -1;
}

/*
 * Producer side of the host-object handoff: QEMU asks us (via
 * GET_SHARED_OBJECT) for the fd behind a UUID we registered with
 * vu_add_shared_object when the host buffer was exported. Return the owning
 * entry's dma_fd; libvhost-user duplicates it over SCM_RIGHTS to the
 * requesting consumer daemon, so we must NOT close it here (its lifetime is
 * the MMAP entry's). Returns -1 if no live buffer carries this UUID.
 *
 * With several devices in one process the UUID may have been registered by a
 * different device than the one QEMU routed this request to (e.g. the decoder
 * produced the buffer, the encoder consumes it), so search every device's
 * media until one resolves the UUID.
 */
static int vu_media_get_shared_object(VuDev *vudev, const unsigned char *uuid)
{
    int fd = vu_ctx_lookup_shared_object(uuid);

    if (fd < 0)
        g_warning("get_shared_object: no live buffer for requested UUID");
    return fd;
}

/*
 * Consumer-side local fast path (media-level local_lookup callback): resolve a
 * UUID against buffers exported by any device in this process, in-address-space,
 * so a consumer (e.g. encoder) never issues the synchronous cross-process RPC
 * for a producer (e.g. decoder) that lives in this same daemon -- which would
 * deadlock the shared main loop. Returns the producer's dma_fd (caller dup's
 * it) or -1 on a local miss (the producer is another process; RPC is safe).
 */
static int vu_media_local_lookup(struct vhost_user_media *m,
                                 const uint8_t uuid[16])
{
    return vu_ctx_lookup_shared_object(uuid);
}

static const VuDevIface vuiface = {
    .get_features = vu_media_get_features,
    .set_features = vu_media_set_features,
    .queue_set_started = vu_media_queue_set_started,
    .get_config = vu_media_get_config,
    .set_config = vu_media_set_config,
    .get_protocol_features = vu_media_get_protocol_features,
    .get_shared_object = vu_media_get_shared_object,
};

/*
 * GLib source to poll V4L2 session fds for events.
 */
static gboolean
session_poll_callback(gint fd, GIOCondition condition,
                                      gpointer user_data)
{
    struct media_session *s = user_data;

    vhost_media_poll_session(s->media, s, condition);
    return G_SOURCE_CONTINUE;
}

/*
 * Register a session's V4L2 fd for polling in the GLib main loop.
 */
static void vu_media_register_poll(struct vhost_user_media *m,
                                   struct media_session *s)
{
    s->poll_source_id = g_unix_fd_add(s->v4l2_fd,
                                      G_IO_IN | G_IO_OUT | G_IO_PRI |
                                      G_IO_ERR | G_IO_HUP,
                                      session_poll_callback, s);
    g_debug("Registered poll for session %u (fd=%d, source=%u)",
            s->id, s->v4l2_fd, s->poll_source_id);
}

static gboolean hangup(gpointer data)
{
    GMainLoop *loop = data;
    g_main_loop_quit(loop);
    return TRUE;
}

/*
 * Bring up one device: allocate its state, take ownership of the connected
 * socket, wire the media callbacks and shared grant handle, and hand the socket
 * to libvhost-user. On success the device is appended to ctx->devices and 0 is
 * returned; on failure everything allocated here is torn down and -1 returned.
 */
static int vu_media_device_new(struct vu_media_ctx *ctx,
                               const char *v4l2_device, int socket_fd)
{
    struct vu_media_device *dev = g_new0(struct vu_media_device, 1);

    dev->ctx = ctx;
    dev->media.v4l2_device = g_strdup(v4l2_device);
    dev->media.send_event = vu_media_send_event;
    dev->media.register_poll = vu_media_register_poll;
    dev->media.local_lookup = vu_media_local_lookup;
    /* Every device shares the one process-wide Xen grant device handle. */
    dev->media.gntdev_handle = ctx->gntdev_handle;

    dev->pending_events = g_queue_new();

    /* Per-device persistent command/response scratch buffers. */
    dev->cmd_buf_size = VU_MEDIA_CMD_BUF_SIZE;
    dev->cmd_buf = g_malloc(dev->cmd_buf_size);
    dev->resp_buf_size = VU_MEDIA_RESP_BUF_SIZE;
    dev->resp_buf = g_malloc(dev->resp_buf_size);

    /*
     * vug_init takes ownership of socket_fd: vu_deinit (via vug_deinit in
     * vu_media_device_free) closes it. The fd is our own (dup'd out of the
     * transient GSocket by the caller), so this is the single owner.
     */
    if (!vug_init(&dev->dev, VHOST_USER_MEDIA_MAX_QUEUES,
                  socket_fd, vu_media_panic, &vuiface)) {
        g_printerr("Failed to initialize libvhost-user-glib for %s\n",
                   v4l2_device);
        close(socket_fd);
        g_free(dev->cmd_buf);
        g_free(dev->resp_buf);
        g_queue_free(dev->pending_events);
        g_free(dev->media.v4l2_device);
        g_free(dev);
        return -1;
    }

    ctx->devices = g_list_append(ctx->devices, dev);
    g_message("vhost-user-media: device=%s ready", v4l2_device);
    return 0;
}

/*
 * Tear down one device: destroy its sessions, drain its grant-release pool,
 * deinit libvhost-user and free its buffers. The shared grant handle lives in
 * the ctx and is closed once, after all devices are gone.
 */
static void vu_media_device_free(struct vu_media_device *dev)
{
    while (dev->media.sessions) {
        struct media_session *s = dev->media.sessions->data;
        vhost_media_destroy_session(&dev->media, s);
    }

    /* Drain and free the grant-release worker pool (waits for pending jobs) */
    if (dev->media.grant_release_pool)
        g_thread_pool_free(dev->media.grant_release_pool, FALSE, TRUE);

    vug_deinit(&dev->dev);

    while (!g_queue_is_empty(dev->pending_events))
        g_free(g_queue_pop_head(dev->pending_events));
    g_queue_free(dev->pending_events);

    g_free(dev->cmd_buf);
    g_free(dev->resp_buf);
    g_free(dev->media.v4l2_device);
    g_free(dev);
}

/*
 * Dup the fd out of a transient GSocket so its lifetime is independent of the
 * wrapper (which is unref'd by the caller, closing its own copy). The returned
 * fd is owned by the caller and ultimately closed by vug_deinit.
 */
static int vu_media_dup_socket_fd(GSocket *socket)
{
    int fd = dup(g_socket_get_fd(socket));

    if (fd < 0)
        g_printerr("Failed to dup socket fd: %m\n");
    return fd;
}

/*
 * Bind + listen a Unix socket path. Returns the listening GSocket (owned by the
 * caller) or NULL on failure. Kept separate from accept so that, with multiple
 * devices, every socket node is created up front: the client (QEMU) connects to
 * all of them before we block in accept. Binding and accepting one device at a
 * time would leave later sockets non-existent, and QEMU would fail to connect.
 */
static GSocket *vu_media_listen(const char *socket_path)
{
    g_autoptr(GError) error = NULL;
    g_autoptr(GSocketAddress) addr = g_unix_socket_address_new(socket_path);
    GSocket *bind_socket =
        g_socket_new(G_SOCKET_FAMILY_UNIX, G_SOCKET_TYPE_STREAM,
                     G_SOCKET_PROTOCOL_DEFAULT, &error);

    if (!bind_socket) {
        g_printerr("Failed to create socket for %s: %s\n",
                   socket_path, error ? error->message : "unknown error");
        return NULL;
    }
    if (!g_socket_bind(bind_socket, addr, false, &error)) {
        g_printerr("Failed to bind to %s: %s\n", socket_path, error->message);
        g_object_unref(bind_socket);
        return NULL;
    }
    if (!g_socket_listen(bind_socket, &error)) {
        g_printerr("Failed to listen on %s: %s\n",
                   socket_path, error->message);
        g_object_unref(bind_socket);
        return NULL;
    }
    g_message("Listening on %s", socket_path);
    return bind_socket;
}

/*
 * Accept a connection on a listening socket. Returns a connected fd (owned by
 * the caller, to be handed to vug_init) or -1 on failure.
 */
static int vu_media_accept(GSocket *listen_socket)
{
    g_autoptr(GError) error = NULL;
    g_autoptr(GSocket) socket = g_socket_accept(listen_socket, NULL, &error);

    if (!socket) {
        g_printerr("Failed to accept connection: %s\n",
                   error ? error->message : "unknown error");
        return -1;
    }
    return vu_media_dup_socket_fd(socket);
}

/*
 * Wrap an inherited (already connected) socket fd. Returns a connected fd
 * (owned by the caller, to be handed to vug_init) or -1 on failure.
 */
static int vu_media_wrap_fd(int socket_fd)
{
    g_autoptr(GError) error = NULL;
    g_autoptr(GSocket) socket = g_socket_new_from_fd(socket_fd, &error);

    if (!socket) {
        g_printerr("Failed to wrap socket fd %d: %s\n",
                   socket_fd, error ? error->message : "unknown error");
        return -1;
    }
    return vu_media_dup_socket_fd(socket);
}

/*
 * Return TRUE if the string array contains a repeated value, storing the index
 * of the first duplicate in *dup_idx. Used to reject ambiguous CLI pairings.
 */
static gboolean strv_has_duplicate(GStrv strv, guint *dup_idx)
{
    guint i, j;

    for (i = 0; strv && strv[i]; i++) {
        for (j = 0; j < i; j++) {
            if (g_strcmp0(strv[i], strv[j]) == 0) {
                *dup_idx = i;
                return TRUE;
            }
        }
    }
    return FALSE;
}

int main(int argc, char *argv[])
{
    g_autoptr(GError) error = NULL;
    g_auto(GStrv) v4l2_devices = NULL;
    g_auto(GStrv) socket_paths = NULL;
    g_auto(GStrv) socket_fds = NULL;
    GSocket **listen_sockets = NULL;
    guint n_devices, n_paths, n_fds, i;

    GOptionEntry entries[] = {
        { "socket-path", 's', 0, G_OPTION_ARG_STRING_ARRAY, &socket_paths,
          "Unix socket path (repeat once per device)", "PATH" },
        { "socket-fd", 'f', 0, G_OPTION_ARG_STRING_ARRAY, &socket_fds,
          "Unix socket fd (repeat once per device)", "FD" },
        { "v4l2-device", 'd', 0, G_OPTION_ARG_STRING_ARRAY, &v4l2_devices,
          "V4L2 device path (repeat once per device)", "PATH" },
        { NULL }
    };

    g_autoptr(GOptionContext) context =
        g_option_context_new("- vhost-user-media daemon");
    g_option_context_add_main_entries(context, entries, NULL);
    if (!g_option_context_parse(context, &argc, &argv, &error)) {
        g_printerr("Option parsing failed: %s\n", error->message);
        exit(EXIT_FAILURE);
    }

    n_devices = v4l2_devices ? g_strv_length(v4l2_devices) : 0;
    n_paths = socket_paths ? g_strv_length(socket_paths) : 0;
    n_fds = socket_fds ? g_strv_length(socket_fds) : 0;

    if (n_devices == 0) {
        g_printerr("Must specify --v4l2-device\n");
        exit(EXIT_FAILURE);
    }

    /*
     * Each device needs exactly one socket, paired by position. Sockets may be
     * given all as paths or all as fds, but not a mix (the pairing would be
     * ambiguous). The socket count must match the device count.
     */
    if (n_paths && n_fds) {
        g_printerr("Use either --socket-path or --socket-fd, not both\n");
        exit(EXIT_FAILURE);
    }
    if (n_paths + n_fds == 0) {
        g_printerr("Must specify --socket-path or --socket-fd\n");
        exit(EXIT_FAILURE);
    }
    if (n_paths + n_fds != n_devices) {
        g_printerr("Number of sockets (%u) must match number of "
                   "--v4l2-device (%u)\n", n_paths + n_fds, n_devices);
        exit(EXIT_FAILURE);
    }

    /*
     * Reject duplicate v4l2 nodes or sockets. Two devices sharing a v4l2 node
     * would open the same hardware twice, and two devices sharing a socket
     * would collide on bind/connect -- both are configuration mistakes that
     * position-paired flags cannot express meaningfully.
     */
    if (strv_has_duplicate(v4l2_devices, &i)) {
        g_printerr("Duplicate --v4l2-device: %s\n", v4l2_devices[i]);
        exit(EXIT_FAILURE);
    }
    if (n_paths && strv_has_duplicate(socket_paths, &i)) {
        g_printerr("Duplicate --socket-path: %s\n", socket_paths[i]);
        exit(EXIT_FAILURE);
    }
    if (n_fds && strv_has_duplicate(socket_fds, &i)) {
        g_printerr("Duplicate --socket-fd: %s\n", socket_fds[i]);
        exit(EXIT_FAILURE);
    }

    memset(&vu_ctx, 0, sizeof(vu_ctx));

    /* Open the Xen grant device once; every device shares this handle. */
    vu_ctx.gntdev_handle = xengnttab_open(NULL, 0);
    if (!vu_ctx.gntdev_handle) {
        g_printerr("Failed to open xengnttab device\n");
        exit(EXIT_FAILURE);
    }

    /* Shared main loop for every device. */
    vu_ctx.loop = g_main_loop_new(NULL, FALSE);
    g_unix_signal_add(SIGHUP, hangup, vu_ctx.loop);
    g_unix_signal_add(SIGINT, hangup, vu_ctx.loop);

    /*
     * Phase 1 (path mode only): bind + listen every socket up front so all
     * socket nodes exist before we block accepting any one of them. A single
     * daemon accepts sequentially, so binding lazily per-device would leave
     * later sockets missing and QEMU's connect to them would fail.
     */
    if (n_paths) {
        listen_sockets = g_new0(GSocket *, n_paths);
        for (i = 0; i < n_paths; i++) {
            listen_sockets[i] = vu_media_listen(socket_paths[i]);
            if (!listen_sockets[i])
                exit(EXIT_FAILURE);
        }
    }

    /* Bring up each device: connect its socket, then init libvhost-user. */
    for (i = 0; i < n_devices; i++) {
        int fd;

        if (n_paths) {
            /*
             * Two-phase for path mode: every socket has already been bound and
             * put into listen state above, so all nodes exist on the
             * filesystem before we block in accept. QEMU connects to all of
             * them; we accept them one at a time here.
             */
            fd = vu_media_accept(listen_sockets[i]);
        } else {
            fd = vu_media_wrap_fd(atoi(socket_fds[i]));
        }

        if (fd < 0) {
            g_printerr("Failed to connect socket for %s\n", v4l2_devices[i]);
            exit(EXIT_FAILURE);
        }
        if (vu_media_device_new(&vu_ctx, v4l2_devices[i], fd) < 0)
            exit(EXIT_FAILURE);
    }

    /* Listening sockets are no longer needed once every device is accepted. */
    for (i = 0; i < n_paths; i++)
        g_object_unref(listen_sockets[i]);
    g_free(listen_sockets);

    g_message("vhost-user-media: %u device(s), entering main loop", n_devices);

    g_main_loop_run(vu_ctx.loop);

    g_message("Main loop exited, cleaning up");
    g_main_loop_unref(vu_ctx.loop);

    /* Tear down every device before closing the shared grant handle. */
    g_list_free_full(vu_ctx.devices, (GDestroyNotify)vu_media_device_free);

    /* Close the shared Xen grant device */
    if (vu_ctx.gntdev_handle)
        xengnttab_close(vu_ctx.gntdev_handle);

    return 0;
}
