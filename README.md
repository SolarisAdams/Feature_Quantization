# Code for input feature quantization

Uses input feature quantization to speedup data loading and save memory.

### Files:

- **compresser** : The main component of BiFeat, include **compression and decompression** code.
- packbits: modules used by scalar quantization. Pack multiple low bitwidth integers into one byte, and its reversed operation
- kmeans: modules used by vector quantization, standard k-means clustering and support of batching.

- examples: an example of training script using compression
  - graphsage
    - model
    - train_compressed : the train script using compression
    - utils
      - compresser : include compress and decompress code
      - load_graph ：load and process datasets
      - packbits : module used by compresser
      - kmeans : module used by compresser




### Parameters:

**compresser.py**

initializing: 

- mode: "vq" or "sq", selecting vector quantization or scalar quantization
- length: 
  - if mode is sq, length mean the number of bit to use, can be 1,2,4...16,32, if length is 32, no quantization would be done.
  - if mode is vq, length mean the number of codebook entries, normally select big numbers like 1024, 2048, 8192, note that larger the length is, the slower vq would be
- width: for vq mode only, the width of each codebook entries, the features would be split into Ceil(feature_dim / width) parts for vector quantization
- device: the device used for compressing, only work for vq, advise on cpu because gpu isn't much faster, it is also used as default device for dequantization

compress:

- features : the features to be quantized
- cache_root : specify the directory where quantized result saved. If cache file exists, compresser will directly use the result.
- dataset_name: dataset name, to generate the filename of cached result
- batch_size: for vq only, read and quantize a batch of nodes each time, only giant datasets needs , doesn't affect training.

decompress:

- compressed_features: features to be dequantized 
- device: device to perform dequantization, features are loaded into device and dequantize.


### Compression algorithm
#### Scalar quantization

Scalar quantization uses a log-uniform quantization method because graph features are mostly near a normal distribution and have most of their values near zero. Log-uniform quantization reduces the overall quantization error and helps keep model accuracy. 

Below, we introduce the feature quantization formula:

$$
Q(x)  =  \begin{cases}
 - \lceil  \frac{Clip(log_{2}(-x))-e_{min}}{e_{max}-e_{min}} *2^{k-1}\rceil , &x<0   \\
\lfloor \frac{Clip(log_{2}x)-e_{min}}{e_{max}-e_{min}} *2^{k-1}\rfloor, &x\ge0     \\
\end{cases}
$$

where $x$ is the original feature, $e_{min}$ and $e_{max}$ are the minimum and maximum values respectively of binary logarithm on non-negative $x$ after clipping outlier values, i.e., $Clip(log_{2}|x|)$.

And the dequantization is the reversed operation of the quantization.

$$
Q^{-1}(q)  =  \begin{cases}
exp2 \left (\frac{(2^{k-1}-0.5 - q)*(e_{max}-e_{min})}{2^{k-1}}+e_{min} \right ), q<2^{k-1}   \\
exp2\left (\frac{(q-2^{k-1}+0.5)*(e_{max}-e_{min})}{2^{k-1}}+e_{min} \right ), q\ge 2^{k-1}     \\
\end{cases}
$$



#### Vector quantization
vector quantization has 2 parameters, width and length. Use OGBN-Papers100M as example, setting width as 16 and length as 256, quantization includes following 2 steps:
1. Divide feature dimension into parts, this will divide each node’s 128-dimension feature into 8 parts.
2. Consider one part at a time, each node’s feature is seen as a vector, we run K-means with k=length=256, and get the cluster centers and each node’s cluster id. 
3. We use a codebook to store these cluster centers, there are in total 8 codebooks each consists of 256 16-dimension vectors. 
4. We use cluster id to represent each node’s feature, so each node uses 8 8-bit integer to store the ids.

During dequantization, using 8 codebooks, we get 8 corresponding 16-dimension cluster centers, and concatenate them to restore the 128-dimension feature.



### Example of using feature quantization:

train_compressed.py shows the usage of compresser, its additional 3 arguments mode, width, length are used to initialize compresser. 

The feature compression is implemented in line 274-275 and the decompression is in line 87. 

Following are 3 examples of using the script to perform full precision training, and feature compressed training with 1-bit scalar quantization and vector quantization.

```sh
python train_compressed.py --dataset ogbn-papers100m --mode sq --length 32
# training without compression
python train_compressed.py --dataset ogbn-papers100m --mode sq --length 1
# training with SQ, quantized into 1 bit(binary feature)
python train_compressed.py --dataset ogbn-papers100m --mode vq --width 16 --length 2048
# training with VQ, advise width and length be (16-2048, 12-1024)
# these are practical setup for large scale graphs,
# for Reddit, compress ratio can be higher, like (64-2048)
```





