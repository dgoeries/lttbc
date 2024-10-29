#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>


static PyObject *downsample(PyObject *self, PyObject *args) {
    int threshold;
    PyObject *x_obj = NULL, *y_obj = NULL;
    PyArrayObject *x_array = NULL, *y_array = NULL;

    if (!PyArg_ParseTuple(args, "OOi", &x_obj, &y_obj, &threshold))
        return NULL;

    if ((!PyArray_Check(x_obj) && !PyList_Check(x_obj)) ||
        (!PyArray_Check(y_obj) && !PyList_Check(y_obj))) {
        PyErr_SetString(PyExc_TypeError, "x and y must be list or ndarray.");
        return NULL;
    }

    x_array = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                                NPY_ARRAY_IN_ARRAY);
    y_array = (PyArrayObject *)PyArray_FROM_OTF(y_obj, NPY_DOUBLE,
                                                NPY_ARRAY_IN_ARRAY);
    if (!x_array || !y_array)
        goto fail;

    if (PyArray_NDIM(x_array) != 1 || PyArray_NDIM(y_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "x and y must be 1-dimensional.");
        goto fail;
    }
    if (!PyArray_SAMESHAPE(x_array, y_array)) {
        PyErr_SetString(PyExc_ValueError, "x and y must have the same shape.");
        goto fail;
    }

    // Declare data length and check if we actually have to downsample!
    const npy_intp data_length = (Py_ssize_t)PyArray_DIM(x_array, 0);
    if (threshold >= data_length || threshold <= 0) {
        // Nothing to do!
        PyObject *result = PyTuple_Pack(2, x_array, y_array);
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        return result;
    }

    // Access the data in the NDArray!
    double *x = (double *)PyArray_DATA(x_array);
    double *y = (double *)PyArray_DATA(y_array);

    // Create an empty output array with shape and dim for the output!
    npy_intp dims[1] = {threshold};
    PyArrayObject *sampled_x =
        (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject *sampled_y =
        (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    // Get a pointer to its data
    double *sampled_x_data = (double *)PyArray_DATA(sampled_x);
    double *sampled_y_data = (double *)PyArray_DATA(sampled_y);

    // The main loop here!
    npy_intp current_index = 0;
    const double every = (double)(data_length - 2) / (threshold - 2);

    Py_ssize_t a = 0;
    Py_ssize_t next_a = 0;

    double max_area_point_x = 0.0;
    double max_area_point_y = 0.0;

    // Always add the first point!
    sampled_x_data[current_index] = npy_isfinite(x[a]) ? x[a] : 0.0;
    sampled_y_data[current_index] = npy_isfinite(y[a]) ? y[a] : 0.0;

    current_index++;
    npy_intp i;
    for (i = 0; i < threshold - 2; ++i) {
        // Calculate point average for next bucket (containing c)
        double avg_x = 0;
        double avg_y = 0;
        npy_intp avg_start = (npy_intp)(floor((i + 1) * every) + 1);
        npy_intp avg_end = (npy_intp)(floor((i + 2) * every) + 1);
        if (avg_end >= data_length) {
            avg_end = data_length;
        }
        npy_intp avg_len = avg_end - avg_start;
        for (npy_intp j = avg_start; j < avg_end; ++j) {
            avg_x += x[j];
            avg_y += y[j];
        }
        avg_x /= avg_len;
        avg_y /= avg_len;

        // Get the range for this bucket
        npy_intp range_start = (npy_intp)(floor(i * every) + 1);
        npy_intp range_end = (npy_intp)(floor((i + 1) * every) + 1);

        // Point a
        double point_a_x = x[a];
        double point_a_y = y[a];

        double max_area = -1.0;
        for (npy_intp k = range_start; k < range_end; ++k) {
            // Calculate triangle area over three buckets
            double area = fabs((point_a_x - avg_x) * (y[k] - point_a_y) -
                               (point_a_x - x[k]) * (avg_y - point_a_y)) *
                          0.5;
            if (area > max_area) {
                max_area = area;
                max_area_point_x = x[k];
                max_area_point_y = y[k];
                next_a = k; // Next a is this b
            }
        }
        // Pick this point from the bucket
        sampled_x_data[current_index] = max_area_point_x;
        sampled_y_data[current_index] = max_area_point_y;
        current_index++;

        // Current a becomes the next_a (chosen b)
        a = next_a;
    }

    // Always add last! Check for finite values!
    double last_a_x = x[data_length - 1];
    double last_a_y = y[data_length - 1];
    sampled_x_data[current_index] = npy_isfinite(last_a_x) ? last_a_x : 0.0;
    sampled_y_data[current_index] = npy_isfinite(last_a_y) ? last_a_y : 0.0;
    PyObject *result = PyTuple_Pack(2, sampled_x, sampled_y);

    // And remove the references!
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_XDECREF(sampled_x);
    Py_XDECREF(sampled_y);

    return result;

fail:
    Py_XDECREF(x_array);
    Py_XDECREF(y_array);
    return NULL;
}

// Method definition object
static PyMethodDef lttbc_methods[] = {
    {"downsample", // The name of the method
     downsample,   // Function pointer to the method implementation
     METH_VARARGS,
     "Compute the largest triangle three buckets (LTTB) algorithm in a C "
     "extension."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef lttbc_module_definition = {
    PyModuleDef_HEAD_INIT, "lttbc",
    "A Python module that computes the largest triangle three buckets "
    "algorithm (LTTB) using C code.",
    -1, lttbc_methods};

// Module initialization
PyMODINIT_FUNC PyInit_lttbc(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&lttbc_module_definition);
}
