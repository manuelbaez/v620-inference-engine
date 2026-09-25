/* tools/inspect.c — GGUF file inspector.
 *
 * Usage:
 *   inspect FILE [--kv] [--validate] [--tensor NAME]
 *
 * Prints a summary by default. --kv dumps all KV pairs (strings truncated
 * to 120 chars, arrays to their first 12 elements). --validate runs
 * gguf_validate and prints the report. --tensor NAME prints the info for
 * one tensor. Exits non-zero with a readable message on missing file or
 * non-GGUF input.
 */
#include "qw/gguf.h"
#include "qw/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void trunc_str(const char *s, int max, char *out, size_t outcap)
{
    size_t n = strlen(s);
    if (n > (size_t)max)
        n = (size_t)max;
    if (n >= outcap)
        n = outcap - 1;
    memcpy(out, s, n);
    out[n] = '\0';
    if (strlen(s) > n)
        strcat(out, "...");
}

static void print_kv(const struct gguf_file *f, size_t i)
{
    const char *key = gguf_get_key(f, i);
    gguf_type t = gguf_get_kv_type(f, i);
    char buf[160];
    if (t == GGUF_STRING) {
        const char *s = gguf_get_val_str(f, i);
        trunc_str(s ? s : "(null)", 120, buf, sizeof(buf));
        printf("  [%2zu] %-40s string   \"%s\"\n", i, key, buf);
    } else if (t == GGUF_ARRAY) {
        gguf_type et = gguf_get_arr_type(f, i);
        uint64_t n = gguf_get_arr_n(f, i);
        printf("  [%2zu] %-40s array<%s> n=%llu",
               i, key, gguf_type_name(et), (unsigned long long)n);
        uint64_t show = n < 12 ? n : 12;
        for (uint64_t j = 0; j < show; j++) {
            if (et == GGUF_STRING) {
                const char *s = gguf_arr_get_str(f, i, j);
                trunc_str(s ? s : "(null)", 40, buf, sizeof(buf));
                printf(" \"%s\"", buf);
            } else if (et == GGUF_UINT32) {
                printf(" %llu", (unsigned long long)gguf_arr_get_u32(f, i, j));
            } else if (et == GGUF_INT32) {
                printf(" %lld", (long long)gguf_arr_get_i32(f, i, j));
            } else if (et == GGUF_FLOAT32) {
                printf(" %.6g", (double)gguf_arr_get_f32(f, i, j));
            } else if (et == GGUF_BOOL) {
                printf(" %s", gguf_arr_get_bool(f, i, j) ? "true" : "false");
            } else if (et == GGUF_ARRAY) {
                printf(" [arr<%s> n=%llu]", gguf_type_name(gguf_arr_get_arr_type(f, i, j)),
                       (unsigned long long)gguf_arr_get_arr_n(f, i, j));
            } else {
                printf(" ?");
            }
        }
        if (n > 12)
            printf(" ...");
        printf("\n");
    } else {
        switch (t) {
        case GGUF_UINT8:
        case GGUF_UINT16:
        case GGUF_UINT32:
        case GGUF_UINT64:
            printf("  [%2zu] %-40s %-8s %llu\n", i, key,
                   gguf_type_name(t),
                   (unsigned long long)(t == GGUF_UINT32
                                            ? (uint64_t)(uint32_t)gguf_get_val_u32(f, i)
                                            : gguf_get_val_u64(f, i)));
            break;
        case GGUF_INT8:
        case GGUF_INT16:
        case GGUF_INT32:
        case GGUF_INT64:
            printf("  [%2zu] %-40s %-8s %lld\n", i, key,
                   gguf_type_name(t),
                   (long long)(t == GGUF_INT32
                                   ? (int64_t)gguf_get_val_i32(f, i)
                                   : gguf_get_val_i64(f, i)));
            break;
        case GGUF_FLOAT32:
            printf("  [%2zu] %-40s %-8s %g\n", i, key, gguf_type_name(t),
                   (double)gguf_get_val_f32(f, i));
            break;
        case GGUF_FLOAT64:
            printf("  [%2zu] %-40s %-8s %g\n", i, key, gguf_type_name(t),
                   gguf_get_val_f64(f, i));
            break;
        case GGUF_BOOL:
            printf("  [%2zu] %-40s %-8s %s\n", i, key, gguf_type_name(t),
                   gguf_get_val_bool(f, i) ? "true" : "false");
            break;
        default:
            printf("  [%2zu] %-40s %s\n", i, key, gguf_type_name(t));
            break;
        }
    }
}

static int print_tensor(const struct gguf_file *f, const char *name)
{
    int idx = gguf_find(f, name);
    if (idx < 0) {
        fprintf(stderr, "tensor '%s' not found\n", name);
        return 1;
    }
    const struct gguf_tensor_info *t = gguf_tensor_by_index(f, (size_t)idx);
    printf("tensor[%d] %s\n", idx, t->name);
    printf("  dims: ");
    for (uint32_t d = 0; d < t->n_dims; d++)
        printf("%llu%s", (unsigned long long)t->dims[d],
               d + 1 < t->n_dims ? " x " : "");
    printf("\n");
    printf("  dtype:  %s\n", gguf_tensor_type_name(t->t));
    printf("  offset: %llu (0x%llx)\n",
           (unsigned long long)t->offset, (unsigned long long)t->offset);
    printf("  nbytes: %llu\n", (unsigned long long)t->nbytes);
    printf("  hw:     %s\n",
           gguf_tensor_supports_hw(f, (size_t)idx) ? "yes" : "no (host-side "
                                                         "conversion needed)");
    return 0;
}

int main(int argc, char **argv)
{
    const char *path = NULL;
    bool want_kv = false, want_val = false;
    const char *tensor = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--kv") == 0)
            want_kv = true;
        else if (strcmp(argv[i], "--validate") == 0)
            want_val = true;
        else if (strcmp(argv[i], "--tensor") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--tensor requires a name\n");
                return 2;
            }
            tensor = argv[++i];
        } else if (path == NULL) {
            path = argv[i];
        } else {
            fprintf(stderr, "unexpected argument '%s'\n", argv[i]);
            return 2;
        }
    }
    if (path == NULL) {
        fprintf(stderr, "usage: inspect FILE [--kv] [--validate] "
                        "[--tensor NAME]\n");
        return 2;
    }

    qw_err err = QW_OK;
    struct gguf_file *f = gguf_open(path, &err);
    if (f == NULL) {
        if (err == QW_ERR_IO)
            fprintf(stderr, "cannot open '%s': %s\n", path,
                    qw_errstr(err));
        else if (err == QW_ERR_FORMAT)
            fprintf(stderr, "'%s' is not a valid GGUF file (%s)\n", path,
                    qw_errstr(err));
        else
            fprintf(stderr, "failed to open '%s': %s\n", path,
                    qw_errstr(err));
        return 1;
    }

    gguf_print_summary(f);

    if (want_kv) {
        printf("KV pairs (%zu):\n", gguf_n_kv(f));
        for (size_t i = 0; i < gguf_n_kv(f); i++)
            print_kv(f, i);
    }
    if (want_val) {
        char report[512] = { 0 };
        bool ok = gguf_validate(f, report, sizeof(report));
        printf("validate: %s: %s\n", ok ? "OK" : "FAIL", report);
    }
    int rc = 0;
    if (tensor != NULL)
        rc = print_tensor(f, tensor);

    gguf_close(f);
    return rc;
}
