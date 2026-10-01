#include <stdio.h>
#include <string.h>
#include <ze_api.h>

int main(void) {
    ze_result_t init = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    printf("zeInit rc=%d (0=success, 0x70000001=UNINITIALIZED)\n", (int)init);

    uint32_t ndrv = 0;
    ze_result_t r = zeDriverGet(&ndrv, NULL);
    printf("zeDriverGet(count) rc=%d ndrivers=%u\n", (int)r, ndrv);

    ze_driver_handle_t drv[16];
    uint32_t nd = 16;
    r = zeDriverGet(&nd, drv);
    printf("zeDriverGet(list) rc=%d ndrivers=%u\n", (int)r, nd);
    if (r != ZE_RESULT_SUCCESS) return 2;

    uint32_t total = 0;
    for (uint32_t i = 0; i < nd; i++) {
        ze_device_handle_t dev[16];
        uint32_t ndv = 16;
        ze_result_t dr = zeDeviceGet(drv[i], &ndv, dev);
        ze_driver_properties_t dp;
        memset(&dp, 0, sizeof dp);
        zeDriverGetProperties(drv[i], &dp);
        printf("  driver[%u] driverVersion=0x%x zeDeviceGet rc=%d DEVICES=%u\n", i,
               (unsigned)dp.driverVersion,
               (int)dr, (dr == ZE_RESULT_SUCCESS) ? ndv : 0);
        if (dr != ZE_RESULT_SUCCESS) continue;
        total += ndv;
        for (uint32_t d = 0; d < ndv; d++) {
            ze_device_properties_t p;
            memset(&p, 0, sizeof p);
            zeDeviceGetProperties(dev[d], &p);
            const unsigned char *u = p.uuid.id;
            printf("    dev[%u] type=%u vendorId=0x%04x deviceId=0x%04x\n",
                   d, (unsigned)p.type, p.vendorId, p.deviceId);
            printf("          uuid=%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
                   u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                   u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
        }
    }
    printf("TOTAL LEVEL ZERO DEVICES: %u\n", total);
    return 0;
}
