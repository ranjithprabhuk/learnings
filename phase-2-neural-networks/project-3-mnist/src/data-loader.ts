import * as tf from '@tensorflow/tfjs-node';

export interface MNISTData {
  trainImages: tf.Tensor;
  trainLabels: tf.Tensor;
  testImages: tf.Tensor;
  testLabels: tf.Tensor;
}

/**
 * MNIST Data Loader
 *
 * Handles loading and preprocessing of the MNIST dataset
 */
export class DataLoader {
  private readonly IMAGE_SIZE = 784; // 28x28
  private readonly NUM_CLASSES = 10;
  private readonly MNIST_IMAGES_URL =
    'https://storage.googleapis.com/cvdf-datasets/mnist/';

  /**
   * Load MNIST dataset
   * Downloads data if not cached, then preprocesses it
   */
  async loadMNIST(): Promise<MNISTData> {
    console.log('Loading MNIST dataset...');

    // Load training data
    const trainImages = await this.loadImages('train-images-idx3-ubyte', 60000);
    const trainLabels = await this.loadLabels('train-labels-idx1-ubyte', 60000);

    // Load test data
    const testImages = await this.loadImages('t10k-images-idx3-ubyte', 10000);
    const testLabels = await this.loadLabels('t10k-labels-idx1-ubyte', 10000);

    console.log('Dataset loaded successfully!');
    console.log(`Training samples: ${trainImages.shape[0]}`);
    console.log(`Test samples: ${testImages.shape[0]}`);

    return {
      trainImages,
      trainLabels,
      testImages,
      testLabels,
    };
  }

  /**
   * Load MNIST images from file
   */
  private async loadImages(filename: string, numSamples: number): Promise<tf.Tensor> {
    const buffer = await this.fetchMNISTData(filename);

    // MNIST image file format:
    // [offset] [type]          [value]          [description]
    // 0000     32 bit integer  0x00000803(2051) magic number
    // 0004     32 bit integer  60000            number of images
    // 0008     32 bit integer  28               number of rows
    // 0012     32 bit integer  28               number of columns
    // 0016     unsigned byte   ??               pixel
    // xxxx     unsigned byte   ??               pixel

    const headerBytes = 16;
    const recordBytes = 28 * 28;

    const headerValues = new Int32Array(buffer, 0, 4);
    const magicNumber = this.readInt32BE(buffer, 0);
    const numImages = this.readInt32BE(buffer, 4);

    if (magicNumber !== 2051) {
      throw new Error(`Invalid magic number: ${magicNumber}`);
    }

    // Extract pixel data
    const imgData = new Float32Array(buffer, headerBytes);

    // Normalize pixel values from [0, 255] to [0, 1]
    const normalizedData = new Float32Array(imgData.length);
    for (let i = 0; i < imgData.length; i++) {
      normalizedData[i] = imgData[i] / 255.0;
    }

    // Return as tensor [numSamples, 784]
    return tf.tensor2d(normalizedData, [numSamples, this.IMAGE_SIZE]);
  }

  /**
   * Load MNIST labels from file
   */
  private async loadLabels(filename: string, numSamples: number): Promise<tf.Tensor> {
    const buffer = await this.fetchMNISTData(filename);

    // MNIST label file format:
    // [offset] [type]          [value]          [description]
    // 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    // 0004     32 bit integer  60000            number of items
    // 0008     unsigned byte   ??               label
    // xxxx     unsigned byte   ??               label

    const headerBytes = 8;
    const magicNumber = this.readInt32BE(buffer, 0);
    const numLabels = this.readInt32BE(buffer, 4);

    if (magicNumber !== 2049) {
      throw new Error(`Invalid magic number: ${magicNumber}`);
    }

    // Extract labels
    const labels = new Uint8Array(buffer, headerBytes, numSamples);

    // Convert to one-hot encoding
    return this.oneHotEncode(Array.from(labels), this.NUM_CLASSES);
  }

  /**
   * Fetch MNIST data file (with caching)
   */
  private async fetchMNISTData(filename: string): Promise<ArrayBuffer> {
    const url = this.MNIST_IMAGES_URL + filename;

    console.log(`Fetching ${filename}...`);

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${filename}: ${response.statusText}`);
    }

    return await response.arrayBuffer();
  }

  /**
   * Read 32-bit big-endian integer
   */
  private readInt32BE(buffer: ArrayBuffer, offset: number): number {
    const view = new DataView(buffer);
    return view.getInt32(offset, false); // false = big-endian
  }

  /**
   * Convert labels to one-hot encoding
   *
   * Example: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
   */
  private oneHotEncode(labels: number[], numClasses: number): tf.Tensor {
    return tf.tidy(() => {
      const oneHot = tf.oneHot(tf.tensor1d(labels, 'int32'), numClasses);
      return oneHot;
    });
  }

  /**
   * Get a random batch of samples
   */
  getBatch(
    images: tf.Tensor,
    labels: tf.Tensor,
    batchSize: number
  ): { images: tf.Tensor; labels: tf.Tensor } {
    return tf.tidy(() => {
      const numSamples = images.shape[0];
      const indices = tf.randomUniform([batchSize], 0, numSamples, 'int32');

      return {
        images: tf.gather(images, indices),
        labels: tf.gather(labels, indices),
      };
    });
  }

  /**
   * Get specific samples by indices
   */
  getSamples(
    images: tf.Tensor,
    labels: tf.Tensor,
    indices: number[]
  ): { images: tf.Tensor; labels: tf.Tensor } {
    return tf.tidy(() => {
      return {
        images: tf.gather(images, tf.tensor1d(indices, 'int32')),
        labels: tf.gather(labels, tf.tensor1d(indices, 'int32')),
      };
    });
  }

  /**
   * Visualize a single MNIST image (as ASCII art)
   */
  visualizeImage(image: tf.Tensor, label?: number): void {
    const pixels = image.dataSync();
    const size = 28;

    console.log(label !== undefined ? `\nLabel: ${label}` : '');
    console.log('─'.repeat(size + 2));

    for (let row = 0; row < size; row++) {
      let line = '│';
      for (let col = 0; col < size; col++) {
        const pixelValue = pixels[row * size + col];

        // Convert pixel value to ASCII character
        if (pixelValue < 0.2) line += ' ';
        else if (pixelValue < 0.4) line += '░';
        else if (pixelValue < 0.6) line += '▒';
        else if (pixelValue < 0.8) line += '▓';
        else line += '█';
      }
      line += '│';
      console.log(line);
    }

    console.log('─'.repeat(size + 2));
  }

  /**
   * Get dataset statistics
   */
  getStatistics(
    images: tf.Tensor,
    labels: tf.Tensor
  ): {
    numSamples: number;
    imageShape: number[];
    labelShape: number[];
    classDistribution: number[];
  } {
    const labelInts = Array.from(labels.argMax(-1).dataSync());
    const classDistribution = new Array(this.NUM_CLASSES).fill(0);

    for (const label of labelInts) {
      classDistribution[label]++;
    }

    return {
      numSamples: images.shape[0],
      imageShape: images.shape,
      labelShape: labels.shape,
      classDistribution,
    };
  }
}
