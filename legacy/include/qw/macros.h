/* qw/macros.h — small portable C17 helpers. */
#ifndef QW_MACROS_H
#define QW_MACROS_H

#define QW_MIN(a, b) ((a) < (b) ? (a) : (b))
#define QW_MAX(a, b) ((a) > (b) ? (a) : (b))

/* Round x up to the next multiple of a. a must be a power of two. */
#define QW_ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((size_t)(a) - 1))

#define QW_ARRAY_COUNT(arr) (sizeof(arr) / sizeof((arr)[0]))

#define QW_UNUSED(x) ((void)(x))

#if defined(__GNUC__) || defined(__clang__)
#define QW_LIKELY(x)   __builtin_expect(!!(x), 1)
#define QW_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define QW_LIKELY(x)   (x)
#define QW_UNLIKELY(x) (x)
#endif

#endif /* QW_MACROS_H */
