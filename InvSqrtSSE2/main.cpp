#include <iostream>
#include <cmath>
#include <emmintrin.h>

float
InvSqrt(float x) {
    float xhalf = 0.5f*x;
    int i = *(int*)&x;
    i = 0x5f3759df - (i >> 1);
    x = *(float*)&i;
    x = x*(1.5f - xhalf*x*x);
    return x;
}

__m128
InvSqrtSSE2(__m128 x) {
    __m128 xhalf = _mm_mul_ps(x, _mm_set1_ps(0.5f));

    x = _mm_castsi128_ps(_mm_sub_epi32(_mm_set1_epi32(0x5f3759df), _mm_srai_epi32(_mm_castps_si128(x), 1)));

    return _mm_mul_ps(x, _mm_sub_ps(_mm_set1_ps(1.5f), _mm_mul_ps(xhalf, _mm_mul_ps(x, x))));
}

int main() {
    float r[4];
    for (int k = 1; k != 101; ++k)
    {
        float x = 1.0f / sqrtf(float(k));
        float y = InvSqrt(float(k));

        __m128 z = _mm_set1_ps(float(k));
        _mm_storeu_ps(r, InvSqrtSSE2(z));

        std::cout << x << " " << y << " " << r[0] << std::endl;
    }

    return 0;
}