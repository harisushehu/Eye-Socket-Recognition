public GaborFeature [] filter( float image[], float mask[], int width, float scales[] )
{
    int nrOrders = 3;
    
    GaborFeature ls[][][] = new GaborFeature[scales.length][nrOrders][];
    float thetas[][] = new float[nrOrders][];
    
    int length = 0;
    
    for(int j = 0; j < scales.length; j++)
    {
        for(int order = 0; order < nrOrders; order++)
        {
            thetas[order] = thetaSet(order);
            ls[j][order] = L(image, mask, width, order, scales[j], thetas[order]);
            
            length += ls[j][order].length;
        }
    }

    GaborFeature Ln[] = new GaborFeature[length];
    
    int index = 0;
    
    for(int j = 0; j < scales.length; j++)
    {
        for(int order = 0; order < nrOrders; order++)
        {
            for(int i = 0; i < ls[j][order].length; i++)
                Ln[index++] = ls[j][order][i];
        }
    }

    ls = (GaborFeature[][][]) null;
    
    return Ln;
}
public float[] thetaSet(int order)
{
    float theta[] = new float[order + 1];
    theta[0] = 0.0F;
    if(order == 1)
        theta[1] = 90F;
    else
    if(order == 2)
    {
        theta[1] = 60F;
        theta[2] = 120F;
    } else
    if(order != 0)
        throw new IllegalArgumentException("order > 2");
    return theta;
}
public GaborFeature [] L(float image[], float mask[], int width, int n, double scale, float theta[])
{
    GaborFeature f[] = new GaborFeature[theta.length];
    float L[][] = new float[theta.length][];
    if(n == 0)
    {
        volume.Kernel1D k0 = new GaussianDerivative(scale, 0);
        L[0] = convolvex(image, mask, width, image.length / width, k0);
        L[0] = convolvey(L[0], width, image.length / width, k0);
        f[0] = new GaborFeature(name(n, scale, 0.0D, ""), L[0]);
    } else
    if(n == 1)
    {
        volume.Kernel1D k0 = new GaussianDerivative(scale, 0);
        volume.Kernel1D k1 = new GaussianDerivative(scale, 1);
        float Lx[] = convolvex(image, mask, width, image.length / width, k1);
        Lx = convolvey(Lx, width, image.length / width, k0);
        float Ly[] = convolvex(image, mask, width, image.length / width, k0);
        Ly = convolvey(Ly, width, image.length / width, k1);
        for(int i = 0; i < theta.length; i++)
        {
            double cth = Math.cos((double)(theta[i] / 180F) * 3.1415926535897931D);
            double sth = Math.sin((double)(theta[i] / 180F) * 3.1415926535897931D);
            float px[] = new float[Lx.length];
            BIJmatrix.mulElements(px, Lx, cth);
            float py[] = new float[Lx.length];
            BIJmatrix.mulElements(py, Ly, sth);
            L[i] = BIJmatrix.addElements(px, py);
            f[i] = new GaborFeature(name(n, scale, theta[i], ""), L[i]);
        }

    } else
    if(n == 2)
    {
        volume.Kernel1D k0 = new GaussianDerivative(scale, 0);
        volume.Kernel1D k1 = new GaussianDerivative(scale, 1);
        volume.Kernel1D k2 = new GaussianDerivative(scale, 2);
        float Lxx[] = convolvex(image, mask, width, image.length / width, k2);
        Lxx = convolvey(Lxx, width, image.length / width, k0);
        float Lxy[] = convolvex(image, mask, width, image.length / width, k1);
        Lxy = convolvey(Lxy, width, image.length / width, k1);
        float Lyy[] = convolvex(image, mask, width, image.length / width, k0);
        Lyy = convolvey(Lyy, width, image.length / width, k2);
        for(int i = 0; i < theta.length; i++)
        {
            double cth = Math.cos((double)(theta[i] / 180F) * 3.1415926535897931D);
            double sth = Math.sin((double)(theta[i] / 180F) * 3.1415926535897931D);
            double c2th = cth * cth;
            double csth = cth * sth;
            double s2th = sth * sth;
            float pxx2[] = new float[Lxx.length];
            BIJmatrix.mulElements(pxx2, Lxx, c2th);
            float pxy2[] = new float[Lxy.length];
            BIJmatrix.mulElements(pxy2, Lxy, 2D * csth);
            float pyy2[] = new float[Lyy.length];
            BIJmatrix.mulElements(pyy2, Lyy, s2th);
            L[i] = BIJmatrix.addElements(pxx2, pxy2);
            BIJmatrix.addElements(L[i], L[i], pyy2);
            f[i] = new GaborFeature(name(n, scale, theta[i], ""), L[i]);
        }

    }
    return f;
}
/**
 * Convolution of plane with 1D separated kernel along the x-axis.
 * The image plane is organized as one 1D vector of width*height.
 * Return the result as a float array. plane is not touched.
 * @param plane the image.
 * @param width the width in pixels of the image.
 * @param height the height of the image in pixels.
 * @param kernel a Kernel1D kernel object.
 * @see Kernel1D
 * @return a float[] with the resulting convolution.
 */
public float [] convolvex(float [] plane, float [] mask, int width, int height, Kernel1D kernel)
{
        float [] result = new float[plane.length];
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
                float d = 0;
                // Around x, convolve over -kernel.halfwidth ..  x .. +kernel.halfwidth.
                for (int k = -kernel.halfwidth; k <= kernel.halfwidth; k++)
                {
                        // Mirror edges if needed.
                        int xi = x+k;
                        int yi = y;
                        if (xi < 0) xi = -xi;
                        else if (xi >= width) xi = 2 * width - xi - 1;
                        if (yi < 0) yi = -yi;
                        else if (yi >= height) yi = 2 * height - yi - 1;
                        if(mask[yi*width+xi]!=0) //sprawdzamy czy maska nie jest zerowa
                        d += plane[yi*width+xi] * kernel.k[k + kernel.halfwidth];
                }
                result[y*width+x] = d;
        }
        return result;
}
/**
 * Convolution of plane with 1D separated kernel along the y-axis.
 * The image plane is organized as one 1D vector of width*height.
 * Return the result as a float array. plane is not touched.
 * @param plane the image.
 * @param width the width in pixels of the image.
 * @param height the height of the image in pixels.
 * @param kernel a Kernel1D kernel object.
 * @see Kernel1D
 * @return a float[] with the resulting convolution.
 */
public float [] convolvey(float [] plane, int width, int height, Kernel1D kernel)
{
        float [] result = new float[plane.length];
        // Convolve in y direction.
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
                float d = 0;
                // Around y, convolve over -kernel.halfwidth ..  y .. +kernel.halfwidth.
                for (int k = -kernel.halfwidth; k <= kernel.halfwidth; k++)
                {
                        // Mirror edges if needed.
                        int xi = x;
                        int yi = y+k;
                        if (xi < 0) xi = -xi;
                        else if (xi >= width) xi = 2 * width - xi - 1;
                        if (yi < 0) yi = -yi;
                        else if (yi >= height) yi = 2 * height - yi - 1;
                        d += plane[yi*width+xi] * kernel.k[k + kernel.halfwidth];
                }
                result[y*width+x] = d;
        }
        return result;
}
public String name(int order, double scale, double theta, String extraText)
{
    if(order == 0)
        return extraText + "L" + order + " scale=" + scale + "p";
    else
        return extraText + "L" + order + "(" + theta + " dgrs) scale=" + scale + "p";
}
public BufferedImage convertToAWT(ImageData data)
{
  ColorModel colorModel = null;
  PaletteData palette = data.palette;
  if (palette.isDirect) {
    colorModel = new DirectColorModel(data.depth, palette.redMask, palette.greenMask,
        palette.blueMask);
    BufferedImage bufferedImage = new BufferedImage(colorModel, colorModel
        .createCompatibleWritableRaster(data.width, data.height), false, null);
    WritableRaster raster = bufferedImage.getRaster();
    int[] pixelArray = new int[3];
    for (int y = 0; y < data.height; y++) {
      for (int x = 0; x < data.width; x++) {
        int pixel = data.getPixel(x, y);
        RGB rgb = palette.getRGB(pixel);
        pixelArray[0] = rgb.red;
        pixelArray[1] = rgb.green;
        pixelArray[2] = rgb.blue;
        raster.setPixels(x, y, 1, 1, pixelArray);
      }
    }
    return bufferedImage;
  } else {
    RGB[] rgbs = palette.getRGBs();
    byte[] red = new byte[rgbs.length];
    byte[] green = new byte[rgbs.length];
    byte[] blue = new byte[rgbs.length];
    for (int i = 0; i < rgbs.length; i++) {
      RGB rgb = rgbs[i];
      red[i] = (byte) rgb.red;
      green[i] = (byte) rgb.green;
      blue[i] = (byte) rgb.blue;
    }
    if (data.transparentPixel != -1) {
      colorModel = new IndexColorModel(data.depth, rgbs.length, red, green, blue,
          data.transparentPixel);
    } else {
      colorModel = new IndexColorModel(data.depth, rgbs.length, red, green, blue);
    }
    BufferedImage bufferedImage = new BufferedImage(colorModel, colorModel
        .createCompatibleWritableRaster(data.width, data.height), false, null);
    WritableRaster raster = bufferedImage.getRaster();
    int[] pixelArray = new int[1];
    for (int y = 0; y < data.height; y++) {
      for (int x = 0; x < data.width; x++) {
        int pixel = data.getPixel(x, y);
        pixelArray[0] = pixel;
        raster.setPixel(x, y, pixelArray);
      }
    }
    return bufferedImage;
  }
}
public ImageData convertToSWT(BufferedImage bufferedImage)
{
  if (bufferedImage.getColorModel() instanceof DirectColorModel)
  {
    DirectColorModel colorModel = (DirectColorModel) bufferedImage.getColorModel();
    PaletteData palette = new PaletteData(colorModel.getRedMask(), colorModel.getGreenMask(),
        colorModel.getBlueMask());
    ImageData data = new ImageData(bufferedImage.getWidth(), bufferedImage.getHeight(),
        colorModel.getPixelSize(), palette);
    WritableRaster raster = bufferedImage.getRaster();
    int[] pixelArray = new int[3];
    for (int y = 0; y < data.height; y++) {
      for (int x = 0; x < data.width; x++) {
        raster.getPixel(x, y, pixelArray);
        int pixel = palette.getPixel(new RGB(pixelArray[0], pixelArray[1], pixelArray[2]));
        data.setPixel(x, y, pixel);
      }
    }
    return data;
  }
  else if (bufferedImage.getColorModel() instanceof IndexColorModel)
  {
    IndexColorModel colorModel = (IndexColorModel) bufferedImage.getColorModel();
    int size = colorModel.getMapSize();
    byte[] reds = new byte[size];
    byte[] greens = new byte[size];
    byte[] blues = new byte[size];
    colorModel.getReds(reds);
    colorModel.getGreens(greens);
    colorModel.getBlues(blues);
    RGB[] rgbs = new RGB[size];
    for (int i = 0; i < rgbs.length; i++) {
      rgbs[i] = new RGB(reds[i] & 0xFF, greens[i] & 0xFF, blues[i] & 0xFF);
    }
    PaletteData palette = new PaletteData(rgbs);
    ImageData data = new ImageData(bufferedImage.getWidth(), bufferedImage.getHeight(),
        colorModel.getPixelSize(), palette);
    data.transparentPixel = colorModel.getTransparentPixel();
    WritableRaster raster = bufferedImage.getRaster();
    int[] pixelArray = new int[1];
    for (int y = 0; y < data.height; y++) {
      for (int x = 0; x < data.width; x++) {
        raster.getPixel(x, y, pixelArray);
        data.setPixel(x, y, pixelArray[0]);
      }
    }
    return data;
  }
  return null;
}
public ImageData convertAWTImageToSWT(java.awt.Image image)
{
  if (image == null) {
      throw new IllegalArgumentException("Null 'image' argument.");
  }
  int w = image.getWidth(null);
  int h = image.getHeight(null);
  if (w == -1 || h == -1) {
      return null;
  }
  BufferedImage bi = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
  Graphics g = bi.getGraphics();
  g.drawImage(image, 0, 0, null);
  g.dispose();
  return convertToSWT(bi);
}
public String [] compare(GaborFeature [] currentEyeSocket, ArrayList<GaborFeature[]> eyeSocketDb, ArrayList<String> nameDb)
{
  String [] result = new String[2]; 
  /*
   * Comparison through Normalized Cross Correlation distance metric
   */
  Vector<Float> distance = new Vector<Float>();
  
  for( GaborFeature [] eyeSocket : eyeSocketDb )
  {
    // iterate through DB
    float dist = 0.0f;
    
    for( int i = 0; i < eyeSocket.length; i++ )
    {
      // iterate through Gabor factors
      float corrDist = 1.0f - ( correlationCoefficient( eyeSocket[i].toVector(), currentEyeSocket[i].toVector(), currentEyeSocket[i].toVector().length ) + 1.0f ) / 2;
      dist += corrDist / eyeSocket.length;
    }
    
    distance.add( dist );
  }

  // select shortest distance
  int shortest = 0;
  float shortDist = (Float) distance.get( 0 );
  
  for( int i = 1; i < distance.size(); i++ )
  {
    if( (Float) distance.get( i ) < shortDist )
    {
      shortest = i;
      shortDist = (Float) distance.get( i );
    }
  }
  
  result[0] = nameDb.get( shortest );
  result[1] = Float.toString( shortDist );
  
  return result;
}

/**
 * @param a is a float[] vector containing a point
 * @param b is a float[] vector containing another point.
 * @return the Normalized Cross Correlation distance between a and b.
 */
// function that returns correlation coefficient. 
public float correlationCoefficient( float [] X, float [] Y, int n )
{  
  float sum_X = 0, sum_Y = 0, sum_XY = 0.0f; 
  float squareSum_X = 0, squareSum_Y = 0.0f; 
 
  for( int i = 0; i < n; i++ ) 
  { 
    // sum of elements of array X. 
    sum_X = sum_X + X[i]; 
 
    // sum of elements of array Y. 
    sum_Y = sum_Y + Y[i]; 
 
    // sum of X[i] * Y[i]. 
    sum_XY = sum_XY + X[i] * Y[i]; 
 
    // sum of square of array elements. 
    squareSum_X = squareSum_X + X[i] * X[i]; 
    squareSum_Y = squareSum_Y + Y[i] * Y[i]; 
  } 
 
  // use formula for calculating correlation  
  // coefficient.
  float divisor = (float)(Math.sqrt((n * squareSum_X - sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y)));
  float corr = 0.0f;
  
  if( divisor > 0 )
    corr = (float)(n * sum_XY - sum_X * sum_Y) / divisor; 
 
  return corr;
}
