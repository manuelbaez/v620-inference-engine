/* src/core/types.c — error strings, dtype sizes, tensor helpers, logging. */
#include "qw/types.h"
#include "qw/macros.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------- errors */
const char *qw_errstr(qw_err e)
{
    switch (e) {
    case QW_OK:             return "ok";
    case QW_ERR_NULL:       return "null pointer";
    case QW_ERR_ALLOC:      return "allocation failure";
    case QW_ERR_IO:         return "i/o error";
    case QW_ERR_FORMAT:     return "bad file format";
    case QW_ERR_UNSUPPORTED: return "unsupported";
    case QW_ERR_DEVICE:     return "device error";
    case QW_ERR_NOMEM:      return "out of memory";
    case QW_ERR_RANGE:      return "value out of range";
    case QW_ERR_NOTIMPL:    return "not implemented";
    default:                return "unknown error";
    }
}

/* --------------------------------------------------------------- dtypes */
int qw_dtype_size(qw_dtype dt)
{
    switch (dt) {
    case QW_F32:    return 4;
    case QW_F16:    return 2;
    case QW_BF16:   return 2;
    case QW_INT8:   return 1;
    case QW_INT4:   return 1; /* packed 2 per byte; use qw_int4_bytes() */
    case QW_UINT8:  return 1;
    case QW_UINT32: return 4;
    default:        return 0;
    }
}

size_t qw_int4_bytes(size_t nelems)
{
    return (nelems + 1) / 2;
}

/* --------------------------------------------------------------- tensor */
qw_err qw_tensor_init(qw_tensor *t, void *data, const int64_t *ne, int n_dims,
                      qw_dtype dt, int dev, bool owns_data)
{
    if (QW_UNLIKELY(t == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY(n_dims < 0 || n_dims > 4))
        return QW_ERR_RANGE;
    if (QW_UNLIKELY(n_dims > 0 && ne == NULL))
        return QW_ERR_NULL;
    if (QW_UNLIKELY((int)dt < 0 || dt >= QW_COUNT))
        return QW_ERR_RANGE;
    if (QW_UNLIKELY(data == NULL && owns_data))
        return QW_ERR_NULL;

    memset(t, 0, sizeof(*t));
    for (int i = 0; i < 4; i++)
        t->ne[i] = (i < n_dims) ? ne[i] : 1;
    t->n_dims    = n_dims;
    t->data      = data;
    t->dt        = dt;
    t->dev       = dev;
    t->owns_data = owns_data;
    return QW_OK;
}

void qw_tensor_free(qw_tensor *t)
{
    if (QW_UNLIKELY(t == NULL))
        return;
    if (t->owns_data && t->data != NULL)
        free(t->data);
    memset(t, 0, sizeof(*t));
}

/* --------------------------------------------------------------- logging */
static int g_qw_log_level = QW_LOG_INFO;

void qw_log_set_level(int level)
{
    g_qw_log_level = level;
}

void qw_log_vprint(int level, const char *file, int line, const char *fmt, ...)
{
    static const char *tag[3] = { "ERROR", "WARN", "INFO" };
    const char *fname = file;
    const char *slash = strrchr(file, '/');
    if (slash != NULL)
        fname = slash + 1;

    if (level > g_qw_log_level)
        return;

    /* fprintf returns an int in C11+; safe to ignore for logging. */
    fprintf(stderr, "%s %s:%d ", tag[level], fname, line);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}
