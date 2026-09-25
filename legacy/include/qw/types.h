/* qw/types.h — core error codes, dtypes, tensor type, logging. Pure C17. */
#ifndef QW_TYPES_H
#define QW_TYPES_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------- errors */
typedef enum qw_err {
    QW_OK = 0,
    QW_ERR_NULL,
    QW_ERR_ALLOC,
    QW_ERR_IO,
    QW_ERR_FORMAT,
    QW_ERR_UNSUPPORTED,
    QW_ERR_DEVICE,
    QW_ERR_NOMEM,
    QW_ERR_RANGE,
    QW_ERR_NOTIMPL,
} qw_err;

const char *qw_errstr(qw_err e);

/* ---------------------------------------------------------------- dtypes */
typedef enum qw_dtype {
    QW_F32,
    QW_F16,
    QW_BF16,
    QW_INT8,
    QW_INT4,
    QW_UINT8,
    QW_UINT32,
    QW_COUNT,
} qw_dtype;

/* Bytes per element. QW_INT4 reports 1 with the convention that int4 values
 * are packed 2-per-byte; use qw_int4_bytes() for actual storage size. */
int qw_dtype_size(qw_dtype dt);

/* Storage bytes for nelems int4 elements (2 per byte, padded to whole byte). */
size_t qw_int4_bytes(size_t nelems);

/* --------------------------------------------------------------- tensor */
typedef struct qw_tensor {
    void    *data;      /* host or device pointer; NULL if !owns_data */
    int64_t  ne[4];     /* extents, fast axis first; unused dims = 1 */
    int      n_dims;    /* 0..4 */
    qw_dtype dt;
    int      dev;       /* -1 = host, >=0 = GPU index */
    bool     owns_data; /* true -> qw_tensor_free frees data */
} qw_tensor;

qw_err qw_tensor_init(qw_tensor *t, void *data, const int64_t *ne, int n_dims,
                      qw_dtype dt, int dev, bool owns_data);
void   qw_tensor_free(qw_tensor *t);

/* --------------------------------------------------------------- logging */
/* Levels: 0 = errors, 1 = warnings, 2 = info (default). */
enum { QW_LOG_ERR = 0, QW_LOG_WARN = 1, QW_LOG_INFO = 2 };

void qw_log_set_level(int level);

#define QW_LOGI(...) qw_log_vprint(2, __FILE__, __LINE__, __VA_ARGS__)
#define QW_LOGW(...) qw_log_vprint(1, __FILE__, __LINE__, __VA_ARGS__)
#define QW_LOGE(...) qw_log_vprint(0, __FILE__, __LINE__, __VA_ARGS__)

void qw_log_vprint(int level, const char *file, int line, const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* QW_TYPES_H */
