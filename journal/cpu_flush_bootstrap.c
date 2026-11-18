/* x86 workers inherit FTZ/DAZ when the constructor runs before numerical imports.
 * Use this for CPU runs that specify cpu_flush_denormal=True.
 */
#include <xmmintrin.h>

__attribute__((constructor)) static void bnn_initialize_floating_point(void) {
    _mm_setcsr(_mm_getcsr() | 0x8040u);
}

unsigned int bnn_cpu_mxcsr(void) {
    return _mm_getcsr();
}
