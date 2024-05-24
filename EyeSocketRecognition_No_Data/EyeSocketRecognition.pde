import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.DirectColorModel;
import java.awt.image.WritableRaster;
import java.awt.image.IndexColorModel;
import java.awt.Graphics2D;
import java.awt.Graphics;
import java.awt.RenderingHints;
import java.awt.Rectangle;
import java.awt.Point;
import it.tidalwave.imageio.arw.*;
import java.io.BufferedWriter;
import java.io.FileWriter;
import javax.activation.MimetypesFileTypeMap;
import gab.opencv.*;
import java.awt.Rectangle;
import java.util.logging.*;

boolean isHistogramImages    = true;
boolean isFeaturesToTextFile = false;

boolean isSingleRecognitionMode = false;
boolean isTotalValidationMode   = true;

int trainDBStart = 1;
int trainDBEnd   = 40;
float trainDBRatio = 0.80f;

OpenCV cvEye;
Rectangle[] eyes;

float socketWidthRatio  = 0.20f;
float socketHeightRatio = 0.05f;

float socketHorShrinkRatio = 0.05f;
float socketVerShrinkRatio = 0.10f;

int imgWidth   = 500;
int imgHeight  = 500;

String trainFilename = "TrainData.txt";
File TrainFile;

String testFilename  = "TestData.txt";
File TestFile;

String path = "MaskedAT&T";

void setup()
{
  LogManager.getLogManager().reset();
            
  if( isFeaturesToTextFile )
  {
    TrainFile = new File( dataPath( trainFilename ) );
    
    if( TrainFile.exists() && TrainFile.isFile() )
    {
      TrainFile.delete();
    }
    
    createFile( TrainFile );
    
    TestFile = new File( dataPath( testFilename ) );
    
    if( TestFile.exists() && TestFile.isFile() )
    {
      TestFile.delete();
    }
    
    createFile( TestFile );
  }
  
  PImage emptyImage = createImage( imgWidth, imgHeight, RGB );
  
  for( int i = 0; i < emptyImage.pixels.length; i++ )
  {
    emptyImage.pixels[i] = color( 0 );
  }
  
  emptyImage.updatePixels();
  
  cvEye = new OpenCV( this, emptyImage );
  cvEye.loadCascade( "haarcascade_mcs_eyepair_big.xml" );
  
  int singlePersonNumber = 1;
  String singleImgPath = sketchPath("data/") + path + "/" + singlePersonNumber + "/001_00000001.png";
  
  ArrayList eyeSocketDatabase = new ArrayList();
  ArrayList nameDatabase = new ArrayList();
  
  if( isTotalValidationMode == true && isSingleRecognitionMode == false )
  {
    trainDatabase( eyeSocketDatabase, nameDatabase, trainDBStart, trainDBEnd );
    testDatabase( eyeSocketDatabase, nameDatabase, trainDBStart, trainDBEnd );
  }
  
  File file = new File( singleImgPath );
  
  PImage img = null;
  
  if( file.isFile() && file.exists() )
  {
    MimetypesFileTypeMap fileType = new MimetypesFileTypeMap();
    fileType.addMimeTypes( "image png jpg jpeg" );
    
    String mimetype = fileType.getContentType( file );
    String type = mimetype.split( "/" )[0];
    
    if( type.equals( "image" ) )
    {
      img = loadImage( singleImgPath );
    }
    else
    {
      println( "File is not an image file!" );
    }
  }
  else
  {
    println( "File is null!" );
  }
  
  if( img != null )
  {
    img.resize( imgWidth, imgHeight );
    img.updatePixels();
    
    size( img.width, img.height );
    
    image( img, 0, 0 );
    
    cvEye.loadImage( img );
    eyes = cvEye.detect();
    
    float roi_x = 0.0f;
    float roi_y = 0.0f;
    float roi_w = 0.0f;
    float roi_h = 0.0f;
    
    for( int i = 0; i < eyes.length; i++ )
    {
      roi_x += eyes[i].x / eyes.length;
      roi_y += eyes[i].y / eyes.length;
      roi_w += eyes[i].width / eyes.length;
      roi_h += eyes[i].height / eyes.length;
    }
    
    if( roi_w > 0 && roi_h > 0 )
    {
      roi_x += (int) ( roi_w * socketHorShrinkRatio );
      roi_y += (int) ( roi_h * socketVerShrinkRatio );
      roi_w -= (int) ( roi_w * socketHorShrinkRatio * 2 );
      roi_h -= (int) ( roi_h * socketVerShrinkRatio * 2 );
      
      PImage roi = createImage( (int) roi_w, (int) roi_h, RGB );
      roi.copy( img, (int) roi_x, (int) roi_y, (int) roi_w, (int) roi_h, 0, 0, (int) roi_w, (int) roi_h );
      roi.updatePixels();
      
      if( isHistogramImages )
      {
        roi.loadPixels();
        roi = getHistogramImage( roi );
        roi.updatePixels();
        img.copy( roi, 0, 0, (int) roi_w, (int) roi_h, (int) roi_x, (int) roi_y, (int) roi_w, (int) roi_h );
        img.updatePixels();
      }
      
      image( img, 0, 0 );
      
      noFill();
      strokeWeight( 2 );
      stroke( 255, 0, 0 );
      rect( (int) roi_x, (int) roi_y, (int) roi_w, (int) roi_h );
      println( "W: " + (int) roi_w + ", H: " + (int) roi_h );
      
      if( isSingleRecognitionMode == true && isTotalValidationMode == false )
      {
        trainDatabase( eyeSocketDatabase, nameDatabase, trainDBStart, trainDBEnd );
        
        if( ( eyeSocketDatabase.size() > 0 ) && ( nameDatabase.size() > 0 ) )
        {
          long startTime = System.currentTimeMillis();
          
          GaborFeature [] gaborVector = getGaborFeatures( roi );
          
          String [] result = compare( gaborVector, eyeSocketDatabase, nameDatabase );
          
          println( "Recognition Result for Person " + singlePersonNumber + " : " + result[0] + ", minDistance: " + result[1] );
          println( "-------------------------------------------" );
          
          long endTime = System.currentTimeMillis();
          
          println( "Elapsed Time: " + ( endTime - startTime ) + " ms." );
          println( "-------------------------------------------" );
        }
        else
        {
          println( "Database is empty!" );
          println( "-------------------------------------------" );
        }
      }
    }
  }
}

GaborFeature [] getGaborFeatures( PImage roi )
{
  roi.loadPixels();
  roi.resize( max( 20, (int) ( imgWidth * socketWidthRatio ) ), max( 20, (int) ( imgHeight * socketHeightRatio ) ) );
  roi.updatePixels();
  
  java.awt.Image i = (java.awt.Image) roi.getImage();
  ImageData imgData = convertAWTImageToSWT( i );
  BufferedImage image = convertToAWT( imgData );
  BufferedImage resizedImage = new BufferedImage( roi.width, roi.height, image.getType() );
  Graphics2D g = resizedImage.createGraphics();
  g.setRenderingHint( RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR );
  g.drawImage( image, 0, 0, roi.width, roi.height, 0, 0, image.getWidth(), image.getHeight(), null );
  g.dispose();
  ij.ImagePlus imga = new ImagePlus( "img", resizedImage );
  ij.process.ImageProcessor ip = imga.getProcessor();
  ij.measure.Calibration cal = imga.getCalibration();
  ip.setCalibrationTable( cal.getCTable() );
  ip = ip.convertToFloat();
  imga = new ImagePlus( "img", ip );
  float[] img_float = bijnum.BIJutil.vectorFromImageStack( imga, 0 ); //image to float[]
  ij.ImagePlus maska = new ImagePlus( "mask", resizedImage );
  ip = maska.getProcessor();
  cal = maska.getCalibration();
  ip.setCalibrationTable( cal.getCTable() );
  ip = ip.convertToFloat();
  maska = new ImagePlus( "mask", ip );
  
  float[] mask = bijnum.BIJutil.vectorFromImageStack( maska, 0 ); //mask to float[]
  float[] scales = { 2, 16 }; //parameter scales
  
  GaborFeature [] gaborFeatures = filter( img_float, mask, imga.getWidth(), scales ); //get Gabor features
  
  return gaborFeatures;
}

PImage getHistogramImage( PImage img )
{
  img.filter( INVERT );
  
  double [] histogram_H    =   new double[256];
  double [] histogram_S    =   new double[256];
  double [] histogram_V    =   new double[256];
  
  for( int i = 0; i < img.pixels.length; i++ )
  {
    color clr = img.pixels[i];
    
    int hue           =   (int) hue  ( clr );
    int saturation    =   (int) saturation( clr );
    int value         =   (int) brightness( clr );
    
    histogram_H[hue]++;
    histogram_S[saturation]++;
    histogram_V[value]++;
  }
  
  for( int i = 0; i < 256; i++ )
  {
    histogram_H[i]   /=   img.pixels.length;
    histogram_S[i]   /=   img.pixels.length;
    histogram_V[i]   /=   img.pixels.length;
  }
  
  PImage histogramImage = createImage( img.width, img.height, RGB );
  
  for( int i = 0; i < img.pixels.length; i++ )
  {
    color clr = img.pixels[i];
    
    int red                  =   (int) red  ( clr );
    int green                =   (int) green( clr );
    int blue                 =   (int) blue ( clr );
    
    int hue                  =   (int) hue  ( clr );
    int saturation           =   (int) saturation( clr );
    int value                =   (int) brightness( clr );
    
    double hue_freq          =   histogram_H[hue];
    double saturation_freq   =   histogram_S[saturation];
    double value_freq        =   histogram_V[value];
    
    int red_geo_mean         =   (int) Math.round( Math.sqrt( red   * ( hue_freq * hue ) ) );
    int green_geo_mean       =   (int) Math.round( Math.sqrt( green * ( saturation_freq * saturation ) ) );
    int blue_geo_mean        =   (int) Math.round( Math.sqrt( blue  * ( value_freq * value ) ) );
    
    histogramImage.pixels[i] = color( red_geo_mean, green_geo_mean, blue_geo_mean );
  }
  
  histogramImage.updatePixels();
  
  return histogramImage;
}

String [] readImagesPath( String dir )
{
  File folder = new File( dir );
  File [] listOfFiles = folder.listFiles();
  
  String [] imgNames = null;
  
  if( listOfFiles != null && listOfFiles.length > 0 )
  {
    imgNames = new String[listOfFiles.length];
    
    int imgCounter = 0;
    
    for( File file : listOfFiles )
    {
      if( file.isFile() && file.exists() )
      {
        MimetypesFileTypeMap fileType = new MimetypesFileTypeMap();
        fileType.addMimeTypes( "image png jpg jpeg" );
        
        String mimetype = fileType.getContentType( file );
        String type = mimetype.split( "/" )[0];
        
        if( type.equals( "image" ) )
        {
          imgNames[imgCounter] = file.getName();
          imgCounter++;
        }
        else
        {
          println( "File is not an image file!" );
        }
      }
      else
      {
        println( "File is null!" );
      }
    }
  }
  else
  {
    println( "Folder is empty!" );
  }
  
  return imgNames;
}

void trainDatabase( ArrayList eyeSocketDatabase, ArrayList nameDatabase, int start, int end )
{ 
  println( "-------------------------------------------" );
  println( "Training started!" );
  
  for( int person = start; person <= end; person++ )
  {
    String [] imgNames = readImagesPath( sketchPath( "data/" + path + "/" + person + "/" ) );
    
    if( imgNames != null && imgNames.length > 0 )
    {
      for( int ii = 0; ii < floor( imgNames.length * trainDBRatio ); ii++ )
      {
        PImage img = loadImage( path + "/" + person + "/" + imgNames[ii] );
        
        if( img != null && img.width > 0 && img.height > 0 )
        {
          img.loadPixels();
          
          size( img.width, img.height );
          image( img, 0, 0 );
          
          img.resize( imgWidth, imgHeight );
          cvEye.loadImage( img );
          eyes = cvEye.detect();
          
          float roi_x = 0.0f;
          float roi_y = 0.0f;
          float roi_w = 0.0f;
          float roi_h = 0.0f;
          
          for( int i = 0; i < eyes.length; i++ )
          {
            roi_x += eyes[i].x / eyes.length;
            roi_y += eyes[i].y / eyes.length;
            roi_w += eyes[i].width / eyes.length;
            roi_h += eyes[i].height / eyes.length;
          }
          
          if( roi_w > 0 && roi_h > 0 )
          {
            roi_x += (int) ( roi_w * socketHorShrinkRatio );
            roi_y += (int) ( roi_h * socketVerShrinkRatio );
            roi_w -= (int) ( roi_w * socketHorShrinkRatio * 2 );
            roi_h -= (int) ( roi_h * socketVerShrinkRatio * 2 );
        
            PImage roi = createImage( (int) roi_w, (int) roi_h, RGB );
            roi.copy( img, (int) roi_x, (int) roi_y, (int) roi_w, (int) roi_h, 0, 0, (int) roi_w, (int) roi_h );
            roi.updatePixels();
            
            if( isHistogramImages )
            {
              roi.loadPixels();
              roi = getHistogramImage( roi );
              roi.updatePixels();
              img.copy( roi, 0, 0, (int) roi_w, (int) roi_h, (int) roi_x, (int) roi_y, (int) roi_w, (int) roi_h );
              img.updatePixels();
            }
            
            noFill();
            stroke( 255, 0, 0 );
            strokeWeight( 2 );
            rect( roi_x, roi_y, roi.width, roi.height );
            
            save( "/train_detect/" + person + "/" + imgNames[ii] + ".png" );
            
            GaborFeature [] gaborVector = getGaborFeatures( roi );
            
            if( gaborVector != null )
            {
              int numOfGaborFeatures = gaborVector.length * gaborVector[0].toVector().length;
              
              //println( "Number of Gabor Features: " + numOfGaborFeatures );
              
              PImage outImg = createImage( (int) ( imgWidth * socketWidthRatio ), round( ( (float) numOfGaborFeatures ) / ( (int) ( imgWidth * socketWidthRatio ) ) ), GRAY );
              outImg.loadPixels();
              
              float [] minGaborFeatures = new float[gaborVector.length];
              float [] maxGaborFeatures = new float[gaborVector.length];
            
              String str = "";
              
              int featureCounter = 0;
              
              for( int g = 0; g < gaborVector.length; g++ )
              {
                float [] gaborFeatures = gaborVector[g].toVector();
                
                minGaborFeatures[g] = min( gaborFeatures );
                maxGaborFeatures[g] = max( gaborFeatures );
                
                float rangeOfGaborFeatures = maxGaborFeatures[g] - minGaborFeatures[g];
                
                for( int f = 0; f < gaborFeatures.length; f++ )
                {
                  float gaborValue = round( ( ( gaborFeatures[f] - minGaborFeatures[g] ) / rangeOfGaborFeatures ) * 100 ) / 100.0f;
                  
                  gaborFeatures[f] = gaborValue;
                  
                  outImg.pixels[featureCounter] = color( round( gaborFeatures[f] * 255 ) );
                  
                  if( isFeaturesToTextFile )
                    str += gaborFeatures[f] + ",";
                    
                  featureCounter++;
                }
              }
              
              outImg.updatePixels();
              outImg.save( "/train_gabor/" + person + "/" + imgNames[ii] + ".png" );
              
              if( isFeaturesToTextFile )
              {
                str += String.valueOf( person );
                appendTextToFile( TrainFile, str );
              }
              
              eyeSocketDatabase.add( gaborVector );
              nameDatabase.add( String.valueOf( person ) );
              
              println( imgNames[ii] + " was registered to database." );
            }
          }
        }
      }
    }
  }
  
  println( "-------------------------------------------" );
  println( "Training finished!" );
  println( "-------------------------------------------" );
}

void testDatabase( ArrayList eyeSocketDatabase, ArrayList nameDatabase, int start, int end )
{
  long startTime = System.currentTimeMillis();
  
  println( "Testing started!" );
  
  int numOfTrueDetections  = 0;
  int numOfFalseDetections = 0;
  
  int counter = 0;
  
  float accuracy = 0.0f;
  
  for( int person = start; person <= end; person++ )
  {
    String [] imgNames = readImagesPath( sketchPath( "data/" + path + "/" + person + "/" ) );
    
    if( imgNames != null && imgNames.length > 0 )
    {
      for( int ii = imgNames.length - 1; ii >= floor( imgNames.length * trainDBRatio ); ii-- )
      {
        PImage img = loadImage( path + "/" + person + "/" + imgNames[ii] );
        
        if( img != null && img.width > 0 && img.height > 0 )
        {
          img.loadPixels();
          
          size( img.width, img.height );
          image( img, 0, 0 );
          
          img.resize( imgWidth, imgHeight );
          cvEye.loadImage( img );
          eyes = cvEye.detect();
          
          float roi_x = 0.0f;
          float roi_y = 0.0f;
          float roi_w = 0.0f;
          float roi_h = 0.0f;
          
          for( int i = 0; i < eyes.length; i++ )
          {
            roi_x += eyes[i].x / eyes.length;
            roi_y += eyes[i].y / eyes.length;
            roi_w += eyes[i].width / eyes.length;
            roi_h += eyes[i].height / eyes.length;
          }
          
          if( roi_w > 0 && roi_h > 0 )
          {
            roi_x += (int) ( roi_w * socketHorShrinkRatio );
            roi_y += (int) ( roi_h * socketVerShrinkRatio );
            roi_w -= (int) ( roi_w * socketHorShrinkRatio * 2 );
            roi_h -= (int) ( roi_h * socketVerShrinkRatio * 2 );
            
            PImage roi = createImage( (int) roi_w, (int) roi_h, RGB );
            roi.copy( img, (int) roi_x, (int) roi_y, (int) roi_w, (int) roi_h, 0, 0, (int) roi_w, (int) roi_h );
            roi.updatePixels();
            
            if( isHistogramImages )
            {
              roi.loadPixels();
              roi = getHistogramImage( roi );
              roi.updatePixels();
              img.copy( roi, 0, 0, (int) roi_w, (int) roi_h, (int) roi_x, (int) roi_y, (int) roi_w, (int) roi_h );
              img.updatePixels();
            }
            
            noFill();
            stroke( 255, 0, 0 );
            strokeWeight( 2 );
            rect( roi_x, roi_y, roi.width, roi.height );
            
            save( "/test_detect/" + person + "/" + imgNames[ii] + ".png" );
            
            if( ( eyeSocketDatabase.size() > 0 ) && ( nameDatabase.size() > 0 ) )
            {
              GaborFeature [] gaborVector = getGaborFeatures( roi );
              
              if( gaborVector != null )
              {
                int numOfGaborFeatures = gaborVector.length * gaborVector[0].toVector().length;
              
                //println( "Number of Gabor Features: " + numOfGaborFeatures );
                
                PImage outImg = createImage( (int) ( imgWidth * socketWidthRatio ), round( ( (float) numOfGaborFeatures ) / ( (int) ( imgWidth * socketWidthRatio ) ) ), GRAY );
                outImg.loadPixels();
                
                float [] minGaborFeatures = new float[gaborVector.length];
                float [] maxGaborFeatures = new float[gaborVector.length];
              
                String str = "";
                
                int featureCounter = 0;
              
                for( int g = 0; g < gaborVector.length; g++ )
                {
                  float [] gaborFeatures = gaborVector[g].toVector();
                
                  minGaborFeatures[g] = min( gaborFeatures );
                  maxGaborFeatures[g] = max( gaborFeatures );
                  
                  float rangeOfGaborFeatures = maxGaborFeatures[g] - minGaborFeatures[g];
                
                  for( int f = 0; f < gaborFeatures.length; f++ )
                  {
                    float gaborValue = round( ( ( gaborFeatures[f] - minGaborFeatures[g] ) / rangeOfGaborFeatures ) * 100 ) / 100.0f;
                  
                    gaborFeatures[f] = gaborValue;
                    
                    outImg.pixels[featureCounter] = color( round( gaborFeatures[f] * 255 ) );
                    
                    if( isFeaturesToTextFile )
                      str += gaborFeatures[f] + ",";
                      
                    featureCounter++;
                  }
                }
                
                outImg.updatePixels();
                outImg.save( "/test_gabor/" + person + "/" + imgNames[ii] + ".png" );
              
                if( isFeaturesToTextFile )
                {
                  str += String.valueOf( person );
                  appendTextToFile( TestFile, str );
                }
                
                String [] result = compare( gaborVector, eyeSocketDatabase, nameDatabase );
                println( "Recognition Result for Person " + person + " : " + result[0] + ", minDistance: " + result[1] );
                
                counter++;
                
                int prediction = Integer.parseInt( result[0] );
                int groundTruth = person;
                
                if( groundTruth == prediction )
                  numOfTrueDetections++;
                else
                  numOfFalseDetections++;
              }
            }
            else
            {
              println( "Database is empty!" );
              println( "-------------------------------------------" );
            }
          }
        }
      }
    }
  }
  
  println( "-------------------------------------------" );
  println( "Testing finished!" );
  
  accuracy = ( (float) numOfTrueDetections ) / ( counter == 0 ? 1 : counter );
  
  println( "-------------------------------------------" );
  println( "Number of Instances: " + counter );
  println( "Number of True Detections: " + numOfTrueDetections );
  println( "Number of False Detections: " + numOfFalseDetections );
  println( "Accuracy: " + accuracy );
  println( "-------------------------------------------" );
  
  long endTime = System.currentTimeMillis();
  
  println( "Elapsed Time: " + ( endTime - startTime ) + " ms." );
  println( "Average Elapsed Time for per Instance: " + ( endTime - startTime ) / ( counter == 0 ? 1 : counter ) + " ms." );
  println( "-------------------------------------------" );
}

void appendTextToFile( File f, String text )
{
  try
  {
    PrintWriter out = new PrintWriter( new BufferedWriter( new FileWriter( f, true ) ) );
    out.println( text );
    out.close();
  }
  catch( IOException e )
  {
    e.printStackTrace();
  }
}

void createFile( File f )
{
  File parentDir = f.getParentFile();
  
  try
  {
    parentDir.mkdirs(); 
    f.createNewFile();
  }
  catch( Exception e )
  {
    e.printStackTrace();
  }
}
