#include <papi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "papi.hpp"

#define FLOPS 1000000
#define THRESHOLD 100000
#define ERROR_RETURN(retval)                                                    \
    {                                                                           \
        fprintf(stderr, "Error %d %s:line %d: \n", retval, __FILE__, __LINE__); \
        exit(retval);                                                           \
    }

float real_time, proc_time, ipc;
long long ins;
float real_time_i, proc_time_i, ipc_i;
long long ins_i;
int retval;
long long sc, ec;

void papi_init()
{
    if ((retval = PAPI_ipc(&real_time_i, &proc_time_i, &ins_i, &ipc_i)) < PAPI_OK)
    {
        printf("Could not initialise PAPI_ipc \n");
        printf("retval: %d\n", retval);
        exit(1);
    }
    // sc = PAPI_get_real_cyc();
}

void papi_stop()
{
    // ec = PAPI_get_real_cyc();

    if ((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK)
    {
        printf("retval: %d\n", retval);
        exit(1);
    }

    printf("Real_time: %f\n Proc_time: %f\n Instructions: %lld\n Cycles: %lf\n IPC: %f\n",
           real_time, proc_time, ins, (double)ipc/(double)ins, ipc);
}