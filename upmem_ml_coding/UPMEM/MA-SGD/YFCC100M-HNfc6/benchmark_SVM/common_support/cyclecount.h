#include <perfcounter.h>
// Timer
typedef struct perfcounter_cycles_t{
    perfcounter_t start;
    perfcounter_t end;
    perfcounter_t end2;

}perfcounter_cycles_t;

void timer_start(perfcounter_cycles_t *cycles){
    cycles->start = perfcounter_get(); // START TIMER
}

uint64_t timer_stop(perfcounter_cycles_t *cycles){
    cycles->end = perfcounter_get(); // STOP TIMER
    cycles->end2 = perfcounter_get(); // STOP TIMER
    return ((uint64_t)(((cycles->end) - (cycles->start)) - ((cycles->end2) - (cycles->end))));
}