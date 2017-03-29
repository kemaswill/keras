<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/utils/generic_utils.py#L16)</span>
### CustomObjectScope

```python
keras.utils.generic_utils.CustomObjectScope()
```

Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

Code within a `with` statement will be able to access custom objects
by name. Changes to global custom objects persist
within the enclosing `with` statement. At end of the `with` statement,
global custom objects are reverted to state
at beginning of the `with` statement.

__Example__


Consider a custom object `MyObject`

```python
with CustomObjectScope({"MyObject":MyObject}):
	layer = Dense(..., W_regularizer="MyObject")
	# save, load, etc. will recognize custom object by name
```

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/utils/io_utils.py#L15)</span>
### HDF5Matrix

```python
keras.utils.io_utils.HDF5Matrix(datapath, dataset, start=0, end=None, normalizer=None)
```

Representation of HDF5 dataset to be used instead of a Numpy array.

__Example__


```python
x_data = HDF5Matrix('input/file.hdf5', 'data')
model.predict(x_data)
```

Providing `start` and `end` allows use of a slice of the dataset.

Optionally, a normalizer function (or lambda) can be given. This will
be called on every slice of data retrieved.

__Arguments__

- __datapath__: string, path to a HDF5 file
- __dataset__: string, name of the HDF5 dataset in the file specified
	in datapath
- __start__: int, start of desired slice of the specified dataset
- __end__: int, end of desired slice of the specified dataset
- __normalizer__: function to be called on data when retrieved

__Returns__

An array-like HDF5 dataset.

----

### to_categorical


```python
to_categorical(y, num_classes=None)
```


Converts a class vector (integers) to binary class matrix.

E.g. for use with categorical_crossentropy.

__Arguments__

- __y__: class vector to be converted into a matrix
(integers from 0 to num_classes).
- __num_classes__: total number of classes.

__Returns__

A binary matrix representation of the input.

----

### normalize


```python
normalize(x, axis=-1, order=2)
```


Normalizes a Numpy array.

__Arguments__

- __x__: Numpy array to normalize.
- __axis__: axis along which to normalize.
- __order__: Normalization order (e.g. 2 for L2 norm).

__Returns__

A normalized copy of the array.

----

### custom_object_scope


```python
custom_object_scope()
```


Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

Convenience wrapper for `CustomObjectScope`.
Code within a `with` statement will be able to access custom objects
by name. Changes to global custom objects persist
within the enclosing `with` statement. At end of the `with` statement,
global custom objects are reverted to state
at beginning of the `with` statement.

__Example__


Consider a custom object `MyObject`

```python
with custom_object_scope({"MyObject":MyObject}):
layer = Dense(..., W_regularizer="MyObject")
# save, load, etc. will recognize custom object by name
```

__Arguments__

- __*args__: Variable length list of dictionaries of name,
class pairs to add to custom objects.

__Returns__

Object of type `CustomObjectScope`.

----

### get_custom_objects


```python
get_custom_objects()
```


Retrieves a live reference to the global dictionary of custom objects.

Updating and clearing custom objects using `custom_object_scope`
is preferred, but `get_custom_objects` can
be used to directly access `_GLOBAL_CUSTOM_OBJECTS`.

__Example__


```python
get_custom_objects().clear()
get_custom_objects()["MyObject"] = MyObject
```

__Returns__

Global dictionary of names to classes (`_GLOBAL_CUSTOM_OBJECTS`).

----

### serialize_keras_object


```python
serialize_keras_object(instance)
```

----

### deserialize_keras_object


```python
deserialize_keras_object(identifier, module_objects=None, custom_objects=None, printable_module_name='object')
```

----

### get_file


```python
get_file(fname, origin, untar=False, md5_hash=None, cache_subdir='datasets')
```


Downloads a file from a URL if it not already in the cache.

Passing the MD5 hash will verify the file after download
as well as if it is already present in the cache.

__Arguments__

- __fname__: name of the file
- __origin__: original URL of the file
- __untar__: boolean, whether the file should be decompressed
- __md5_hash__: MD5 hash of the file for verification
- __cache_subdir__: directory being used as the cache

__Returns__

Path to the downloaded file

----

### convert_all_kernels_in_model


```python
convert_all_kernels_in_model(model)
```


Converts all convolution kernels in a model from Theano to TensorFlow.

Also works from TensorFlow to Theano.

__Arguments__

- __model__: target model for the conversion.

----

### plot_model


```python
plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True)
```
