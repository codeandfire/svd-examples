This project has two examples of the SVD.

The first is a text example, taken from an example in the original paper on Latent Semantic Analysis [(Deerwester et. al. (1990))](http://lintool.github.io/UMD-courses/CMSC723-2009-Fall/readings/Deerwester_etal_1990.pdf). 
There are 9 sample documents (listed in `deerwester.txt`), of which the first 5 belong to the topic of human-computer interaction, and the remaining 4 belong to the topic of graph theory.
When using `k = 2` singular values, documents and terms from these two categories get neatly separated out into two perpendicular directions.
With different values of `k`, we no longer get the same result.

Run this using:
```
$ cargo run -- text
```
By default `k` is 2. To use a different value of `k`, check out the help text of this command:
```
$ cargo run -- text --help
```

The second example uses a 1024x768 image of a raccoon face from Scipy.
Using successive values of `k`, we compute the rank `k` approximation of this image.
The resulting images show us that the first few singular values capture the broad, macro features of the image, while the remaining singular values capture the finer, minor details.

This example is adapted from <https://github.com/Ramaseshanr/anlp/blob/master/SVDImage.ipynb>.

Run this using:
```
$ cargo run -- image
```
To change the `k` values used, take a look at the help text again:
```
$ cargo run -- image --help
```

To run tests, please use:
```
$ cargo test
```
