/*
 * Vhost User library
 *
 * Copyright 2024 NXP
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or
 * later.  See the COPYING file in the top-level directory.
 */

#include <glib.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdarg.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <endian.h>
#include <fcntl.h>
#include "libvhost-user.h"

#include <xenctrl.h>
#include <xenforeignmemory.h>
#include <xengnttab.h>

/*
 * Process-wide cache of foreign-mapped guest RAM regions.
 *
 * A single vhost-user-media daemon now drives several virtio-media devices,
 * one VuDev per device.  Every VuDev negotiates the guest memory table and
 * asks to foreign-map the guest RAM, but for a given DomU that is the *same*
 * physical range (same domid/gpa/size) for all of them.  Mapping it once per
 * device multiplies the Dom0 foreign-map pressure by the device count and, at
 * four devices, exhausts it (xenforeignmemory_map2 fails -> vring addr NULL ->
 * panic -> DomU hangs).
 *
 * Map each unique (domid, gpa, size) region exactly once and hand the same VA
 * back to every device, reference-counted.  The mapping (and its Xen handle)
 * is torn down only when the last device drops its reference.
 */
typedef struct VuXenForeignMap {
    uint32_t domainid;
    uint64_t addr;
    uint64_t size;
    void *ramptr;
    xenforeignmemory_handle *handle;
    unsigned int refcount;
} VuXenForeignMap;

static GSList *vu_xen_foreignmaps;
static GMutex vu_xen_foreignmap_lock;

void *vu_xen_foreignmap(uint32_t domainid, uint64_t addr, uint64_t size)
{
    xenforeignmemory_handle * vu_foreignmap_handle = NULL;
    void* ramptr;
    int nb_pfn = size >> XC_PAGE_SHIFT;
    xen_pfn_t *pfns;
    xen_pfn_t address_index = addr >> XC_PAGE_SHIFT;
    int *err;
    int i;
    VuXenForeignMap *entry;
    GSList *l;

    g_mutex_lock(&vu_xen_foreignmap_lock);

    /* Reuse an existing mapping of the same guest range if present. */
    for (l = vu_xen_foreignmaps; l; l = l->next) {
        entry = l->data;
        if (entry->domainid == domainid && entry->addr == addr &&
            entry->size == size) {
            entry->refcount++;
            ramptr = entry->ramptr;
            g_mutex_unlock(&vu_xen_foreignmap_lock);
            DPRINT("xen foreign map reuse domain[%d]: gpa %016"PRIx64
                   " size %016"PRIx64" va addr %p refcount %u\n",
                   domainid, addr, size, ramptr, entry->refcount);
            return ramptr;
        }
    }

    pfns = g_new0(xen_pfn_t, nb_pfn);
    err = g_new0(int, nb_pfn);

    for (i = 0; i < nb_pfn; i++) {
        pfns[i] = address_index + i;
    }

    vu_foreignmap_handle = xenforeignmemory_open(NULL,0);
    if (!vu_foreignmap_handle) {
        DPRINT("xenforeignmemory_open failure\n");
        g_free(pfns);
        g_free(err);
        g_mutex_unlock(&vu_xen_foreignmap_lock);
        return NULL;
    }

    ramptr = xenforeignmemory_map2(vu_foreignmap_handle, domainid, 0,
                                   PROT_READ | PROT_WRITE, 0, nb_pfn, pfns, err);
    if (!ramptr) {
        DPRINT("xenforeignmemory_map2 fail\n");
        xenforeignmemory_close(vu_foreignmap_handle);
        g_free(pfns);
        g_free(err);
        g_mutex_unlock(&vu_xen_foreignmap_lock);
        return NULL;
    }

    entry = g_new0(VuXenForeignMap, 1);
    entry->domainid = domainid;
    entry->addr = addr;
    entry->size = size;
    entry->ramptr = ramptr;
    entry->handle = vu_foreignmap_handle;
    entry->refcount = 1;
    vu_xen_foreignmaps = g_slist_prepend(vu_xen_foreignmaps, entry);

    g_free(pfns);
    g_free(err);
    g_mutex_unlock(&vu_xen_foreignmap_lock);

    DPRINT("xen foreign map domain[%d]: gpa %016"PRIx64" size %016"PRIx64" va addr %p\n",
           domainid, addr, size, ramptr);

    return ramptr;
}

/*
 * Drop one reference to a shared foreign mapping identified by its VA.  The
 * mapping is unmapped and its Xen handle closed only when the final reference
 * is released.  Returns true if it handled the address (i.e. it was a shared
 * foreign mapping); the caller then must not munmap() it itself.
 */
static bool vu_xen_foreign_unmap(void *ramptr, uint64_t size)
{
    VuXenForeignMap *entry;
    GSList *l;

    g_mutex_lock(&vu_xen_foreignmap_lock);

    for (l = vu_xen_foreignmaps; l; l = l->next) {
        entry = l->data;
        if (entry->ramptr == ramptr) {
            if (--entry->refcount > 0) {
                g_mutex_unlock(&vu_xen_foreignmap_lock);
                DPRINT("xen foreign unmap defer va %p refcount %u\n",
                       ramptr, entry->refcount);
                return true;
            }
            vu_xen_foreignmaps = g_slist_remove(vu_xen_foreignmaps, entry);
            g_mutex_unlock(&vu_xen_foreignmap_lock);

            munmap(ramptr, size);
            xenforeignmemory_close(entry->handle);
            DPRINT("xen foreign unmap release va %p size %016"PRIx64"\n",
                   ramptr, size);
            g_free(entry);
            return true;
        }
    }

    g_mutex_unlock(&vu_xen_foreignmap_lock);
    return false;
}

/*
 * Unmap a guest RAM region previously mapped for this VuDev.  Under Xen the
 * region is a reference-counted shared foreign mapping, so this releases our
 * reference and only tears the mapping down when the last device drops it.
 * Regions that were not shared foreign mappings fall back to a plain munmap().
 * Installed as the region's unmap callback at map time, so the generic core
 * tears the region down without ever referencing Xen code.
 */
static void vu_xen_unmap_region(void *ramptr, uint64_t size)
{
    if (!vu_xen_foreign_unmap(ramptr, size)) {
        munmap(ramptr, size);
    }
}

bool vu_xen_add_mem_reg(VuDev *dev, VhostUserMsg *vmsg)
{
    int i;
    uint32_t domid = vmsg->payload.xenmemreg.xen_domainid;
    VhostUserMemoryRegion m = vmsg->payload.xenmemreg.region, *msg_region = &m;
    VuDevRegion *dev_region = &dev->regions[dev->nregions];

    if (vmsg->size < VHOST_USER_MEM_REG_SIZE) {
        DPRINT("VHOST_USER_ADD_MEM_REG requires a message size of at "
               "least %zu bytes and only %d bytes were received",
               VHOST_USER_MEM_REG_SIZE, vmsg->size);
        return false;
    }

    if (dev->nregions == VHOST_USER_MAX_RAM_SLOTS) {
        DPRINT("failing attempt to hot add memory via "
               "VHOST_USER_ADD_MEM_REG message because the backend has "
               "no free ram slots available");
        return false;
    }


    DPRINT("Adding region: %u %d %x\n", dev->nregions, vmsg->request, vmsg->flags);
    DPRINT("    guest_phys_addr: 0x%016"PRIx64"\n",
           msg_region->guest_phys_addr);
    DPRINT("    memory_size:     0x%016"PRIx64"\n",
           msg_region->memory_size);
    DPRINT("    userspace_addr   0x%016"PRIx64"\n",
           msg_region->userspace_addr);
    DPRINT("    mmap_offset      0x%016"PRIx64"\n",
           msg_region->mmap_offset);

    dev->xen_domainid = domid;
    dev_region->gpa = msg_region->guest_phys_addr;
    dev_region->size = msg_region->memory_size;
    dev_region->qva = msg_region->userspace_addr;
    dev_region->mmap_offset = msg_region->mmap_offset;
    dev_region->mmap_addr = (uint64_t)vu_xen_foreignmap(domid, dev_region->gpa, dev_region->size);
    if (!dev_region->mmap_addr) {
        DPRINT("foreignmap dom[%d] gpa %"PRId64" %"PRId64"\n",
               domid, dev_region->gpa, dev_region->size);
        return false;
    }
    dev_region->unmap = vu_xen_unmap_region;

    DPRINT("dev->max_queues %d\n", dev->max_queues);
    for (i = 0; i < dev->max_queues; i++) {
        if (dev->vq[i].vring.desc) {
            if (map_ring(dev, &dev->vq[i])) {
                DPRINT("remapping queue %d for new memory region", i);
            }
        }
    }

    DPRINT("Successfully added new region\n");
    dev->nregions++;
    /* false means no reply, true means reply. TODO: update to reply*/
    return false;
}

void* vu_open_grantdev(void)
{
    void* xen_gntdev_handle = NULL;

    xen_gntdev_handle = xengnttab_open(NULL, 0);
    if (!xen_gntdev_handle) {
        DPRINT("can't open gnttab device\n");
        return NULL;
    }


    return xen_gntdev_handle;
}

void vu_release_grantdev(void* handler)
{
    if (handler)
        xengnttab_close(handler);

}

int vu_export_grant_resource_pages(void* handler, uint32_t domid, uint32_t count,uint32_t* refs)
{
   uint32_t dmabuf_fd;
   int ret;

   ret = xengnttab_dmabuf_exp_from_refs((xengnttab_handle *)handler, domid,
                                      O_CLOEXEC | O_RDWR, count,
                                      refs,
                                      &dmabuf_fd);

    DPRINT("export resource ret %d,fd %d",ret,dmabuf_fd);

    return dmabuf_fd;
}

int vu_release_grant_resource_pages(void* handler, int fd)
{
    return xengnttab_dmabuf_exp_wait_released((xengnttab_handle *)handler,
                                                fd, 1000);
}