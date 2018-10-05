package com.rscnn.network;

/**
 * Detection Result, contains the classes, bounding boxes and the confidence score.
 */

public class DetectResult {
    private int x1;
    private int y1;
    private int x2;
    private int y2;
    private String cls;
    private float prob;
    private int index;

    public DetectResult(String cls, float prob) {
        this.x1 = this.x2 = this.y1 = this.y2 = 0;
        this.cls = cls;
        this.prob = prob;
    }

    public DetectResult(int x1, int y1, int x2, int y2, String cls, float prob) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.cls = cls;
        this.prob = prob;
    }

    /**
     * x coordinates of the bounding box top left corner.
     * @return
     */
    public int getX1() {
        return x1;
    }

    /**
     * y coordinates of the bounding box top left corner.
     * @return
     */
    public int getY1() {
        return y1;
    }

    /**
     * x coordinates of the bounding box bottom right corner.
     * @return
     */
    public int getX2() {
        return x2;
    }

    /**
     * y coordinates of the bounding box bottom right corner.
     * @return
     */
    public int getY2() {
        return y2;
    }

    /**
     * class name of this object.
     * @return
     */
    public String getCls() {
        return cls;
    }

    /**
     * confidence score of the result, 0 to 1.0.
     * @return
     */
    public float getProb() {
        return prob;
    }

    /**
     * may be you want to get the class index of the classes list
     * @return class index , from 0 to total class number - 1.
     */
    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }
}
