import java.util.Vector;

public class GaborFeature
{
  public float vector[];
  public String name;

  public GaborFeature( String name, float vector[] )
  {
      this.vector = vector;
      this.name = name;
  }

  public String toString()
  {
      return name;
  }

  public int length()
  {
      return vector.length;
  }
  
  public void unitvar()
  {
      vector = BIJstats.unitvar(vector);
  }

  public float [] toVector()
  {
      return vector;
  }

}
