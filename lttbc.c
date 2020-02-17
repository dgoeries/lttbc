#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


static PyObject* downsample(PyObject *self, PyObject *args) {
    int threshold;
    PyObject *x_obj, *y_obj;

    if (!PyArg_ParseTuple(args, "OOi", &x_obj, &y_obj, &threshold))
        return NULL;

    // Interpret the input objects as numpy arrays
    PyArrayObject *x_array = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_array = (PyArrayObject *)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (x_array == NULL || y_array == NULL) {
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        return NULL;
    }
    const int N = (int)PyArray_DIM(x_array, 0);
    const int M = (int)PyArray_DIM(y_array, 0);
    // Dimension check for both input arrays
    if (N != M) {
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_RuntimeError, "X and Y must have the same dimension!");
    }

    // Declare data length and check if we actually have to downsample!
    const int data_length = N;
    if (threshold >= data_length || threshold == 0) {
        // Nothing to do!
        PyObject *value = Py_BuildValue("OO", x_array, y_array);
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        return value;
        }

    // Access the data in the NDArray!
    double *x = (double*)PyArray_DATA(x_array);
    double *y = (double*)PyArray_DATA(y_array);

    // Create an empty output array with shape and dim for the output!
    npy_intp dims[1];
    dims[0] = threshold;
    PyArrayObject *sampled_x = (PyArrayObject *)PyArray_Empty(1, dims,
        PyArray_DescrFromType(NPY_DOUBLE), 0);
    PyArrayObject *sampled_y = (PyArrayObject *)PyArray_Empty(1, dims,
        PyArray_DescrFromType(NPY_DOUBLE), 0);
    // Get access to its data
    double *sampled_x_data = (double*)PyArray_DATA(sampled_x);
    double *sampled_y_data = (double*)PyArray_DATA(sampled_y);

    // The main loop here!
    int sampled_index = 0;
    const float every = (float)(data_length - 2) / (threshold - 2);

    int a = 0;
    int next_a = 0;

    float max_area_point_x = 0.0;
    float max_area_point_y = 0.0;

    // Always add the first point!
    if (npy_isfinite(x[a])) {
        sampled_x_data[sampled_index] = x[a];
    }
    else {
         sampled_x_data[sampled_index] = 0;
    }
    if (npy_isfinite(y[a])) {
        sampled_y_data[sampled_index] = y[a];
    }
    else {
         sampled_y_data[sampled_index] = 0;
    }
    sampled_index++;
    int i;
    for (i = 0; i < threshold - 2; ++i) {
        // Calculate point average for next bucket (containing c)
        float avg_x = 0;
        float avg_y = 0;
        int avg_range_start = (int)(floor((i + 1)* every) + 1);
        int avg_range_end = (int)(floor((i + 2) * every) + 1);
        if (avg_range_end >= data_length){
            avg_range_end = data_length;
        }
        int avg_range_length = avg_range_end - avg_range_start;

        for (;avg_range_start < avg_range_end; avg_range_start++){
            avg_x += x[avg_range_start];
            avg_y += y[avg_range_start];
        }
        avg_x /= avg_range_length;
        avg_y /= avg_range_length;

        // Get the range for this bucket
        int range_offs = (int)(floor((i + 0) * every) + 1);
        int range_to = (int)(floor((i + 1) * every) + 1);

        // Point a
        float point_a_x = x[a];
        float point_a_y = y[a];

        float max_area = -1.0;
        for (; range_offs < range_to; range_offs++){
            // Calculate triangle area over three buckets
            float area = fabs((point_a_x - avg_x) * (y[range_offs] - point_a_y) - (point_a_x - x[range_offs]) * (avg_y - point_a_y)) * 0.5;
            if (area > max_area){
                max_area = area;
                max_area_point_x = x[range_offs];
                max_area_point_y = y[range_offs];
                next_a = range_offs; // Next a is this b
            }
        }
        // Pick this point from the bucket
        sampled_x_data[sampled_index] = max_area_point_x;
        sampled_y_data[sampled_index] = max_area_point_y;
        sampled_index++;

        // Current a becomes the next_a (chosen b)
        a = next_a;
    }

    // Always add last! Check for finite values!
    float last_a_x = x[data_length - 1];
    float last_a_y = y[data_length - 1];
    if (npy_isfinite(last_a_x)) {
        sampled_x_data[sampled_index] = last_a_x;
    }
    else {
         sampled_x_data[sampled_index] = 0;
    }
    if (npy_isfinite(last_a_y)) {
        sampled_y_data[sampled_index] = last_a_y;
    }
    else {
        sampled_y_data[sampled_index] = 0;
    }

    // Provide our return value
    PyObject *value = Py_BuildValue("OO", sampled_x, sampled_y);

    // And remove the references!
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_XDECREF(sampled_x);
    Py_XDECREF(sampled_y);

    return value;
}

// Method definition object
static PyMethodDef lttb_methods[] = {
    {
        "downsample", // The name of the method
        downsample, // Function pointer to the method implementation
        METH_VARARGS, // Flags indicating special features of this method
        "Compute the largest triangle three buckets (LTTB) algorithm in a C extension." // Contents of this method's docstring
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef lttb_module_definition = {
    PyModuleDef_HEAD_INIT,
    "lttb",
    "A Python module that computes the largest triangle three buckets algorithm (LTTB) using C code.",
    -1,
    lttb_methods
};

// Module initialization
PyMODINIT_FUNC PyInit_lttbc(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&lttb_module_definition);
}