import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from keras import backend as K
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np
import gzip
from pathlib import Path
import os
import shutil

tfd = tfp.distributions

def NB_cont_mixture(out):
    epsilon = 1E-24
    total_count_1 = tf.squeeze(tf.math.exp(out[:,0:1])) + epsilon
    logit_1 = tf.squeeze(out[:,1:2])
    rate_1 = tf.squeeze(tf.math.softplus(out[:, 2:3])) + epsilon

    return tfd.Mixture(
    cat=tfd.Categorical(logits=tf.concat([-out[:, 3:4], out[:, 3:4]], axis=-1)),
    components=[
      tfd.Exponential(rate=rate_1),
      tfd.NegativeBinomial(total_count=total_count_1, logits=logit_1, require_integer_total_count=False)
  ])

def NB_cont(out):
    epsilon = 1E-24
    total_count = tf.squeeze(tf.math.exp(out[:,0:1])) + epsilon
    logits = tf.squeeze(out[:,1:2])
    
    return tfd.NegativeBinomial(total_count=total_count, logits=logits, require_integer_total_count=False)

def log_pearson_r(y_true, y_pred):
    epsilon = 10e-5
    x = tf.math.log(y_true + 1)/tf.math.log(2.0)
    y = tf.math.log(y_pred + 1)/tf.math.log(2.0)
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))

    r = r_num / (r_den + epsilon)

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r

def pearson_r(y_true, y_pred):
    epsilon = 10e-5
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))

    r = r_num / (r_den + epsilon)

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r

def spearman_r(y_true, y_pred):
    x = y_true
    y = y_pred
    return ( tf.py_function(spearmanr, [tf.cast(y, tf.float32), 
                       tf.cast(x, tf.float32)], Tout = tf.float32) )

def log_squared_dist(y, p_y):
    epsilon = 1
    return (tf.math.log(y+epsilon) - tf.math.log(p_y+epsilon))**2

def r_square(y_true, y_pred):
    epsilon = 1E-24
    rss =  tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    tss = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=-1)), axis=-1)
    return (1 - rss/(tss + epsilon))


def log_r_square(y_true, y_pred):
    epsilon = 1E-24
    lyt = tf.math.log(y_true+1)/tf.math.log(2.0)
    lyp = tf.math.log(y_pred+1)/tf.math.log(2.0)
    rss =  tf.reduce_sum(tf.square(lyt - lyp), axis=-1)
    tss = tf.reduce_sum(tf.square(lyt - tf.reduce_mean(lyt, axis=-1)), axis=-1)
    return (1 - rss/(tss + epsilon))

def accuracy(y_true, y_pred):
    global expression_median
    gt = tf.logical_and(tf.greater_equal(y_true, expression_median), tf.greater_equal(y_pred, expression_median))
    lt = tf.logical_and(tf.less_equal(y_true, expression_median), tf.less_equal(y_pred, expression_median))
    corrects = tf.logical_or(gt, lt)
    corrects = tf.cast(corrects, dtype=tf.float32)
    return tf.reduce_mean(corrects)

# Given an array of cdfs, calculate a percentile from 0 to 1 (for each cdf in parallel). Uses binary search.
def percentileFromCDFGen(cdf, percentile, size):
    tol = 1E-4

    # Initialize a
    a = np.zeros(size)
    dontJustOutputZero = np.where(cdf(a) < percentile)

    # Initialize b such that cdf(b) > percentile for all array elements
    b = np.zeros(size)
    guessPercentiles = cdf(b).numpy()

    b[guessPercentiles < percentile] += 1
    while tf.reduce_any(guessPercentiles < percentile):
        b[guessPercentiles < percentile] *= 2
        guessPercentiles = cdf(b).numpy()

    # Now iterate through until we reach tolerance
    c = (a + b)/2
    guessPercentiles = cdf(c).numpy()
    cnt = 0
    while tf.reduce_mean(tf.abs(guessPercentiles[dontJustOutputZero] - percentile)) >= tol:
        a[guessPercentiles < percentile] = c[guessPercentiles < percentile]

        b[guessPercentiles > percentile] = c[guessPercentiles > percentile]

        c = (a + b)/2
        guessPercentiles = cdf(c).numpy()
        cnt += 1
    
    return c

# Given a cumulative density function, calculate a percentile between 0 and 1. Uses binary search.
def percentileFromCDF(cdf, percentile):
    # Implement some sort of inary search - We know cdf is monotonic increasing.
    tol = 1E-4
    
    # First check if cdf(0) > percentile. If so, we can just return 0.
    if cdf(0).numpy() >= percentile:
        return 0
    
    # if percentile > cdf(0), find a such that cdf(a) < percentile
    a = 0.1
    while cdf(a).numpy() >= percentile:
        a = a/2
    
    # Next find b such that cdf(b) > percentile
    b = 0.1
    while cdf(b).numpy() <= percentile:
        b = b*2
    
    # Now we want c such that cdf(c) = percentile. We know cdf(a) < percentile < cdf(b), so binary search!
    c = (a+b)/2
    
    while np.abs(percentile - cdf(c).numpy()) > tol:
        c = (a+b)/2
        if cdf(c).numpy() < percentile:
            a = c
        elif cdf(c).numpy() > percentile:
            b = c
        else:
            return c
    return c


def negloglik(y, p_y):
    lp = -p_y.log_prob(y)

    # return lp + 0.3*(tf.math.log(y+1) - tf.math.log(p_y.mean()+1))**2
    return lp


def unitLoss(y, p_y):
    return p_y

def motifKernelInit(shape, dtype=None):
    PSSMKernels = tf.squeeze(np.load(gzip.GzipFile('data/PSSMKernels.npy.gz', 'r')))
    # tf.print(tf.transpose(PSSMKernels,perm=[2, 1, 0] ), summarize=35)
    PSSMKernels = tf.cast(PSSMKernels, dtype=tf.float32)
    # tfBiases = np.load(gzip.GzipFile('data/tfBiasArray.npy.gz', 'r'))
    # PSSMKernels = PSSMKernels * (1 - tfBiases)
    # tf.print(tf.transpose(PSSMKernels,perm=[2, 1, 0]), summarize=35)
    #return K.reverse(PSSMKernels, axes=[0, 1])
    return PSSMKernels

def POLII_init(shape, dtype=None):
    POLIIKernels = tf.squeeze(np.load(gzip.GzipFile('data/POLIIKernels.npy.gz', 'r')))
    POLIIKernels = tf.cast(POLIIKernels, dtype=tf.float32)
    return POLIIKernels


def motifMatchingInit(shape, dtype=None):
    tfMotifMatchArray = np.load(gzip.GzipFile('data/tfMotifMatchArray.npy.gz', 'r'))
    return tfMotifMatchArray


def xorshiftNextIndex(promoterDataIdx, sampleIdx):
    i = np.array(promoterDataIdx*17382 + sampleIdx, dtype=np.uint32)
    a = np.array(13, dtype=np.uint32)
    b = np.array(17, dtype=np.uint32)
    c = np.array(5, dtype=np.uint32)

    while True:
        i ^= i << a
        i ^= i >> b
        i ^= i << c
        # if i // 17382 < 182522:
        if i // 17382 < 19786:
            break

    nextPromoterDataIdx = i // 17382
    nextSampleIdx = i % 17382

    return nextPromoterDataIdx, nextSampleIdx


def generateExamples(set):
    global promoterDataIdx, sampleIdx, val_promIdxs, val_smplIdxs, test_promIdxs, test_smplIdxs

    if set == 1: # Train
        for i in range(NUM_TRAIN):
            promoterDataIdx, sampleIdx = xorshiftNextIndex(promoterDataIdx, sampleIdx)

            # If this promoter index or sample index is in our validation set, generate another.
            while promoterDataIdx in val_promIdxs or promoterDataIdx in test_promIdxs or sampleIdx in val_smplIdxs or sampleIdx in test_smplIdxs:
                promoterDataIdx, sampleIdx = xorshiftNextIndex(promoterDataIdx, sampleIdx)
            
            yield ((creSeqArr[promoterDataIdx], acgtSeqArr[promoterDataIdx], chrArr[promoterDataIdx], tssArr[promoterDataIdx], tfExpressArr[sampleIdx], cellTypeArr[sampleIdx], hmArr[promoterDataIdx]), (expressionArr[promoterDataIdx, sampleIdx]))

    elif set == 2: # Validation
        for i in range(NUM_VAL_TRANSCRIPTS * NUM_VAL_SAMPLES):
            promIdx = val_promIdxs[i%NUM_VAL_TRANSCRIPTS]
            smplIdx = val_smplIdxs[i//NUM_VAL_TRANSCRIPTS]

            yield ((creSeqArr[promIdx], acgtSeqArr[promIdx], chrArr[promIdx], tssArr[promIdx], tfExpressArr[smplIdx], cellTypeArr[smplIdx], hmArr[promIdx]), (expressionArr[promIdx, smplIdx]))

    elif set == 3: # Test
        for i in range(NUM_TEST_TRANSCRIPTS * NUM_TEST_SAMPLES):
            promIdx = test_promIdxs[i%NUM_TEST_TRANSCRIPTS]
            smplIdx = test_smplIdxs[i//NUM_TEST_TRANSCRIPTS]

            yield ((creSeqArr[promIdx], acgtSeqArr[promIdx], chrArr[promIdx], tssArr[promIdx], tfExpressArr[smplIdx], cellTypeArr[smplIdx], hmArr[promIdx]), (expressionArr[promIdx, smplIdx]))

# Because of the way we generate examples, this is merely the number of examples in each epoch.
# However, subsequent epochs will have different examples. In other words, trained for long enough,
# the model will eventually loop through all 330 million examples (except ones in test or val set),
# regardless of what number is set here.
NUM_TRAIN = 1280000//10
NUM_VAL_TRANSCRIPTS = 64
NUM_VAL_SAMPLES = 64
NUM_TEST_TRANSCRIPTS = 256
NUM_TEST_SAMPLES = 256
BATCH_SIZE = 128

# Define global variables which hold dataset indices, from which we will generate examples. This seed can be modified to get different train data from the set.
np.random.seed(3)
seed = np.random.randint(0, 2**32, dtype=np.uint32)
promoterDataIdx, sampleIdx = xorshiftNextIndex(seed // 17382, seed % 17382)

# Generate the indices for our validation set. This seed cannot be modified else validation & test indices will be shuffled into the training data.
np.random.seed(0)
valAndTestProm = np.random.choice(19746, NUM_VAL_TRANSCRIPTS+NUM_TEST_TRANSCRIPTS, replace=False)
valAndTestSmpl = np.random.choice(17382, NUM_VAL_SAMPLES+NUM_TEST_SAMPLES, replace=False)

val_promIdxs = valAndTestProm[0:NUM_VAL_TRANSCRIPTS]
test_promIdxs = valAndTestProm[NUM_VAL_TRANSCRIPTS:NUM_VAL_TRANSCRIPTS+NUM_TEST_TRANSCRIPTS]

val_smplIdxs = valAndTestSmpl[0:NUM_VAL_SAMPLES]
test_smplIdxs = valAndTestSmpl[NUM_VAL_SAMPLES:NUM_VAL_SAMPLES+NUM_TEST_SAMPLES]

ds_train = tf.data.Dataset.from_generator(generateExamples, args=[1], output_signature=((
    tf.TensorSpec(shape=(2823, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(2823, 4), dtype=tf.float32),
    tf.TensorSpec(shape=(24, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.float32),
    tf.TensorSpec(shape=(2753, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(54, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(2823, 3, 7), dtype=tf.float32)),
    (tf.TensorSpec(shape=(), dtype=tf.float32)

)))
ds_train = ds_train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_val = tf.data.Dataset.from_generator(generateExamples, args=[2], output_signature=((
    tf.TensorSpec(shape=(2823, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(2823, 4), dtype=tf.float32),
    tf.TensorSpec(shape=(24, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.float32),
    tf.TensorSpec(shape=(2753, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(54, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(2823, 3, 7), dtype=tf.float32)),
    (tf.TensorSpec(shape=(), dtype=tf.float32)

)))
ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_test = tf.data.Dataset.from_generator(generateExamples, args=[3], output_signature=((
    tf.TensorSpec(shape=(2823, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(2823, 4), dtype=tf.float32),
    tf.TensorSpec(shape=(24, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.float32),
    tf.TensorSpec(shape=(2753, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(54, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(2823, 3, 7), dtype=tf.float32)),
    (tf.TensorSpec(shape=(), dtype=tf.float32)

)))
ds_test = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

data_dir = Path("data")
if (not (data_dir / "creSeqArr_gene.npy.gz").is_file() or not (data_dir / "acgtSeqArr_gene.npy.gz").is_file() or
    not (data_dir / "tfExpressArr.npy.gz").is_file() or not (data_dir / "cellTypeArr.npy.gz").is_file() or
    not (data_dir / "expressionArr_gene.npy.gz").is_file()):

    dl_manager = tfds.download.DownloadManager(download_dir="~/tensorflow_datasets/downloads")
    path = dl_manager.download_and_extract("https://archive.org/download/ibbme-tr3b-data/IBBME_TR3B_Data.zip")

    file_names = os.listdir(path)
    if not (data_dir).exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    for file_name in file_names:
        shutil.move(os.path.join(path, file_name), data_dir)

creSeqArr = np.load(gzip.GzipFile(data_dir / 'creSeqArr_gene.npy.gz', 'r'))
acgtSeqArr = np.load(gzip.GzipFile(data_dir / 'acgtSeqArr_gene.npy.gz', 'r'))
chrArr = np.load(gzip.GzipFile(data_dir / 'chrArr_gene.npy.gz', 'r'))
tssArr = np.load(gzip.GzipFile(data_dir / 'tssArr_gene.npy.gz', 'r'))
tfExpressArr = np.load(gzip.GzipFile(data_dir / 'tfExpressArr.npy.gz', 'r'))
cellTypeArr = np.load(gzip.GzipFile(data_dir / 'cellTypeArr.npy.gz', 'r'))
hmArr = np.load(gzip.GzipFile(data_dir / 'hmArr.npy.gz', 'r'))
expressionArr = np.load(gzip.GzipFile(data_dir / 'expressionArr_gene.npy.gz', 'r'))

expression_median = np.median(np.ndarray.flatten(expressionArr))
#print("Expression median: " + str(expression_median))
#print("expression mean: " + str(np.mean(expressionArr)))

creSeqArr /= 1000

tfChrArr = np.load(gzip.GzipFile(data_dir / 'tfChrArr.npy.gz', 'r'))
tfTssArr = np.load(gzip.GzipFile(data_dir / 'tfTssArr.npy.gz', 'r'))

tssMean = tf.reduce_mean(tssArr)
tssArr = np.squeeze(tssArr)
tssArr /= tssMean
tfTssArr /= tssMean
#print(tssMean)
hmArr /= np.mean(hmArr)


# Layer to compute distance factor and multiply TF levels by this distance factor
class FactorByDiffusion(layers.Layer):
  def __init__(self):
      super(FactorByDiffusion, self).__init__()

  def build(self, input_shape):  # Create the state of the layer (weights)
    b_init = tf.keras.initializers.constant(value=0)
    r_init = tf.keras.initializers.constant(value=0)

    self.w = tf.Variable(
        name='fbd_b',
        initial_value=b_init(shape=(),
                             dtype='float32'),
        trainable=True)
    self.r = tf.Variable(
        name='fbd_r',
        initial_value=r_init(shape=(), dtype='float32'),
        trainable=True)
    
  def call(self, tfLevels, chr, tss):  # Defines the computation from inputs to outputs
    global tfChrArr, tfTssArr
    sameChrSelect = tf.einsum('bc, tc->bt', chr[:, :, 0], tf.squeeze(tf.cast(tfChrArr, dtype=tf.float32)))
    x = tf.abs(tss - tf.cast(tfTssArr, dtype=tf.float32))
 
    e_r = tf.exp(self.r)
    # tf.print(x.sahpe)
    slideFactor = tf.exp(-e_r*x)

    slidingComponent = sameChrSelect * tf.transpose(slideFactor)
    # tf.print(slidingComponent.shape)
    diffusionFactors = slidingComponent + tf.exp(self.w)
    # tf.print(diffusionFactors.shape)
    out = tfLevels[:, :, 0] * diffusionFactors
    # tf.print(out.shape)
    tf.debugging.assert_all_finite(self.r, "r went too big")
    tf.debugging.assert_all_finite(self.w, "w went too big")
    tf.debugging.assert_all_finite(x, "x went too big :()")
    tf.debugging.assert_all_finite(out, "out went too big")

    return out

# Simple layer that merely scales and adds a bias to last dimension.
class ScaleAndBiasLayer(tf.keras.layers.Layer):
    def __init__(self, bias_on=True, *args, **kwargs):
        self.bias_on = bias_on
        super(ScaleAndBiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        if self.bias_on:
            self.bias = self.add_weight('bias',
                                        shape=input_shape[-1],
                                        initializer='zeros',
                                        trainable=True)
        self.gamma = self.add_weight('gamma',
                                    shape=input_shape[-1],
                                    initializer='ones',
                                    trainable=True)
    def call(self, x):
        if self.bias_on:
            return x * self.gamma + self.bias
        else:
            return x * self.gamma

# Instead of zeroing elements of the DNA sequence like dropout typically does, we 
# wish to instead replace zeroed rows with [0.25, 0.25, 0.25, 0.25]—this represents
# no knowledge of which base pair exists here.
def dropoutFix(x):
    norm_x = x/tf.reduce_max(x)

    rowwiseSum = tf.reduce_sum(norm_x, axis=2, keepdims=True)
    out = (norm_x + 0.25) - 0.25 * rowwiseSum
    return out

# Set hyperparameters
dropout=0.2
d_model=128

input_cres = layers.Input(shape=(2823, 1))
input_acgt = layers.Input(shape=(2823, 4))
input_chr = layers.Input(shape=(24, 1))
input_tss = layers.Input(shape=())
input_tflevels = layers.Input(shape=(2753, 1))
input_celltype = layers.Input(shape=(54, 1))
input_hm = layers.Input(shape=(2823, 3, 7))

dropout_acgt = layers.Dropout(0.1)(input_acgt)
dropout_acgt = layers.Lambda(lambda x: dropoutFix(x))(dropout_acgt)

celltype_select = layers.Flatten()(input_celltype)
celltype_select = layers.Dense(7*4)(celltype_select)
celltype_select = layers.Reshape((7, 4))(celltype_select)

dropout_hm = layers.Dropout(0.2)(input_hm)
hm_selected = layers.Lambda(lambda x: tf.einsum('bijk, bkl->bijl', x[0], x[1]))([dropout_hm, celltype_select])
hm_selected = layers.Reshape((2823, 4*3))(hm_selected)

core = layers.Lambda(lambda x: tf.pad(x, [[0, 0], [9, 9], [0, 0]], 'CONSTANT', constant_values=0.25))(dropout_acgt)
core = layers.Conv1D(13, kernel_size=19, padding='valid', kernel_initializer=POLII_init, trainable=False)(core)
#core = layers.Concatenate(axis=2)([core, input_cres, hm_selected])
#core = layers.Concatenate(axis=2)([core, input_cres])
core = layers.Concatenate(axis=2)([hm_selected, core])
#core = hm_selected

core0 = layers.Cropping1D((2703, 0))(core)
core1 = layers.Cropping1D((2463, 120))(core)
core2 = layers.Cropping1D((1983, 360))(core)
core3 = layers.Cropping1D((0, 840))(core)

#core0 = layers.MaxPooling1D(pool_size=2, strides=2)(core0)
core1 = layers.MaxPooling1D(pool_size=2, strides=2)(core1)
core2 = layers.MaxPooling1D(pool_size=6, strides=6)(core2)
core3 = layers.MaxPooling1D(pool_size=18, strides=18)(core3)

core = layers.Concatenate(axis=1)([core3, core2, core1, core0]) # 246 x 14
core = ScaleAndBiasLayer(bias_on=True)(core)
core = layers.ReLU()(core)

z = layers.Lambda(lambda x: tf.pad(x, [[0, 0], [17, 17], [0, 0]], 'CONSTANT', constant_values=0.25))(dropout_acgt)
z = layers.Conv1D(1072, kernel_size=35, padding='valid', kernel_initializer=motifKernelInit, trainable=False)(z)

z0 = layers.Cropping1D((2703, 0))(z)
z1 = layers.Cropping1D((2463, 120))(z)
z2 = layers.Cropping1D((1983, 360))(z)
z3 = layers.Cropping1D((0, 840))(z)

#z0 = layers.MaxPooling1D(pool_size=2, strides=2)(z0)
z1 = layers.MaxPooling1D(pool_size=2, strides=2)(z1)
z2 = layers.MaxPooling1D(pool_size=6, strides=6)(z2)
z3 = layers.MaxPooling1D(pool_size=18, strides=18)(z3)

z = layers.Concatenate(axis=1)([z3, z2, z1, z0]) # 216 x 1237

#tfact = FactorByDiffusion()(input_tflevels, input_chr, input_tss)
tfact = layers.Flatten()(input_tflevels)
tfact = layers.Dense(1072, kernel_initializer=motifMatchingInit, trainable=False)(tfact)
tfact = ScaleAndBiasLayer(bias_on=False)(tfact)
#tfact = layers.BatchNormalization(gamma_initializer=tf.keras.initializers.constant(value=10))(tfact)
#tfact = layers.Dropout(0.05)(tfact)

#z = layers.BatchNormalization()(z)
z = ScaleAndBiasLayer(bias_on=True)(z)
z = layers.ReLU()(z)
z = layers.Multiply()([z, tfact])
#z = layers.BatchNormalization()(z)

z = layers.Dense(d_model-25)(z)
z = layers.SpatialDropout1D(dropout)(z)

z = layers.Concatenate(axis=2)([z, core])
#z = layers.Dropout(0.2)(z)

for i in range(12):
    if i <= 1:
        ks = 7
    elif i <= 3:
        ks = 5
    else:
        ks = 3
    z_ = z

    z = layers.LayerNormalization()(z)
    z = layers.LocallyConnected1D(32, kernel_size=1, activation='relu')(z)
    z = layers.Dropout(dropout)(z)
    z = layers.Add()([z, layers.Dense(32)(z_)])
    z = layers.LayerNormalization()(z)
    z = layers.Conv1D(d_model, kernel_size=ks, activation='relu', padding='same')(z)
    z = layers.Dropout(dropout)(z)
    z = layers.Add()([z, z_])
    if i % 2 == 0:
      z = layers.MaxPooling1D(pool_size=2, strides=2)(z)



z = layers.LayerNormalization()(z)
z = layers.Flatten()(z)

z = layers.Dense(4)(z)

z7 = tfp.layers.DistributionLambda(NB_cont_mixture)(z)
model = tf.keras.Model(inputs=[input_cres, input_acgt, input_chr, input_tss, input_tflevels, input_celltype, input_hm], outputs=[z7])
model.summary()

losses = {"distribution_lambda": negloglik}
lossWeights = {"distribution_lambda": 1}

model.compile(tf.keras.optimizers.Adam(learning_rate=0.0003), loss=losses, loss_weights=lossWeights, metrics=['mean_absolute_error', pearson_r, log_pearson_r, spearman_r, r_square, log_r_square, log_squared_dist, accuracy])
#model.load_weights("conv_localConnect.hdf5")

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "conv_localConnect.hdf5",
    monitor = 'loss',
    save_best_only = False,
    save_weights_only=True,
    save_freq='epoch',
)

#history = model.fit(ds_train, epochs=50, validation_data=ds_val, callbacks=[checkpoint])

# Creating predicted vs true plots
y_true_arr = []
y_pred_arr = []
cellType_idx_arr = []
for example in ds_val:
    ret = model(example[0])

    exampleBatchDim = ret.tensor_distribution.batch_shape_tensor().numpy()[0]

    y_pred_medians = percentileFromCDFGen(ret.tensor_distribution.cdf, 0.50, exampleBatchDim)

    y_true_arr = tf.concat([y_true_arr, example[1].numpy()], axis=-1)
    y_pred_arr = tf.concat([y_pred_arr, y_pred_medians], axis=-1)    


max_y = max([tf.reduce_max(y_true_arr).numpy(), tf.reduce_max(y_pred_arr).numpy()])
print(max_y)

r2 = r_square(y_true_arr, y_pred_arr).numpy()
lr2 = log_r_square(y_true_arr, y_pred_arr).numpy()
p_r = pearson_r(y_true_arr, y_pred_arr).numpy()
lp_r = log_pearson_r(y_true_arr, y_pred_arr).numpy()
s_r = spearman_r(y_true_arr, y_pred_arr).numpy()
acc = accuracy(y_true_arr, y_pred_arr).numpy()
print('Sample Level Data')
print('pearson_r = ' + str(p_r) + ', log_pearson_r = ' + str(lp_r) + ', spearman_r = ' + str(s_r) + ', r2 = ' + str(r2) + ', lr2 = ' + str(lr2) + ', acc = ' + str(acc))

x = np.linspace(0, np.log(max_y + 1)/np.log(10.0), 1000)
plt.plot(x, x, alpha=0.25)
plt.scatter(np.log(y_true_arr + 1)/np.log(10.0), np.log(y_pred_arr + 1)/np.log(10.0), s=0.3)
plt.xlabel('True expression level (Log transformed TPM)')
plt.ylabel('Predicted expression level (Log transformed TPM)')
plt.title('True vs predicted, R^2 = ' + str(lr2))
plt.show()


# Plot some predictions on validation set
cnt = 0
for example in ds_val:
    ret = model(example[0])
    
    for i in range(10):
        dist = ret.tensor_distribution[i]
        x = np.linspace(0, 100, 1000)
        #plt.plot(x, dist.cdf(x).numpy())
        trueval = example[1][i].numpy()
        #plt.show()
        print("**** Negative Binomial Distribution  - For true value {:.3f}".format(trueval) + " ****")
        print("** Parameters:")
        #print("Count (n): {:.3f}".format(dist.parameters['total_count'].numpy()))
        #print("Probs (p): {:.3f}".format(tf.math.sigmoid(dist.parameters['logits']).numpy()))
        print("** Statistics:")
        print("Mean     : {:.3f}".format(dist.mean().numpy()))
        print("Var      : {:.3f}".format(dist.variance().numpy()))
        print("Std. Dev.: {:.3f}".format(dist.stddev().numpy()))
        print("Median   : {:.3f}".format(percentileFromCDF(dist.cdf, 0.5)))
        print("** Confidence Intervals:")
        print("90% CI   : [{:.3f}".format(percentileFromCDF(dist.cdf, 0.1)) + ", {:.3f}".format(percentileFromCDF(dist.cdf, 0.9)) + "]")
        print("95% CI   : [{:.3f}".format(percentileFromCDF(dist.cdf, 0.05)) + ", {:.3f}".format(percentileFromCDF(dist.cdf, 0.95)) + "]")
        print("99% CI   : [{:.3f}".format(percentileFromCDF(dist.cdf, 0.01)) + ", {:.3f}".format(percentileFromCDF(dist.cdf, 0.99)) + "]")

        # Plot pdf on interval from 0 to 99.9%
        x = np.linspace(0, percentileFromCDF(dist.cdf, 0.999), 5000)
        zer = np.linspace(0, 0, 5000)
        plt.plot(x, dist.prob(x).numpy())
        plt.plot(x, zer, color='gray')
        plt.plot(dist.mean().numpy(), dist.prob(dist.mean().numpy()).numpy(), color='r', marker='x')
        plt.axvline(x = trueval, color = 'g', linestyle = '--', linewidth=1)
        plt.plot(trueval, dist.prob(trueval).numpy(), color='k', marker='o')
        plt.title('Probability Density Function for true value: ' + str(example[1][i].numpy()))
        plt.show()
    cnt += 1
    if cnt > 32:
        break

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss over training')
plt.ylabel('Loss')
plt.xlabel('Training steps ('r'$ \times 10^{4}$)')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
# summarize history for pearson
plt.plot(history.history['pearson_r'])
plt.plot(history.history['val_pearson_r'])
plt.plot(history.history['log_pearson_r'])
plt.plot(history.history['val_log_pearson_r'])
plt.title('Pearson correlation over training')
plt.ylabel('Correlation')
plt.xlabel('Training steps ('r'$ \times 10^{4}$)')
plt.legend(['Pearson r', 'val Pearson r', 'log transformed Pearson r', 'val log transformed Pearson r'], loc='upper left')
plt.show()
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy over training')
plt.ylabel('Accuracy')
plt.xlabel('Training steps ('r'$ \times 10^{4}$)')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()