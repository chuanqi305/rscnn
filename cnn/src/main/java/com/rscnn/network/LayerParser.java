package com.rscnn.network;

import android.content.res.AssetManager;
import android.renderscript.RenderScript;
import android.util.ArraySet;
import android.util.Log;

import com.rscnn.layers.Data;
import com.rscnn.layers.Input;
import com.rscnn.utils.LogUtil;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.lang.reflect.Array;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

class LayerParser {
    private static final String TAG = "LayerParser";
    private RenderScript renderScript;

    private String baseDir = "/";
    private Set<String> failedMethod;

    private AssetManager asset;
    private boolean useAsset = false;

    private static final String LAYER_CLASS_PACKAGE = "com.rscnn.layers";

    /*filter some parameter never used in test phase */
    private static final String[] LAYER_PARAMETER_FITER =
            {"lr_mult","decay_mult","weight_filler_type","weight_filler_std",
                    "bias_filler_type","bias_filler_value", "engine", "axis"
            };

    LayerParser(RenderScript renderScript) {
        this.renderScript = renderScript;
        this.failedMethod = new ArraySet<>();
    }

    private byte[][] getLayerParameterFromFile(String layerName) throws IOException {
        String paramPrefix = baseDir+"/"+layerName.replace("/","_");
        int i = 0;
        List<byte[]> list = new ArrayList<>();
        String filePath = "";
        while(true){
            filePath = paramPrefix + "-" + i + ".dat";
            byte[] array;
            if(useAsset){
                try {
                    InputStream inputStream = asset.open(filePath);
                    array = new byte[inputStream.available()];
                    inputStream.read(array);
                }
                catch (IOException e) {
                    break;
                }
            }
            else {
                File f = new File(filePath);
                if(!f.exists() || f.length()==0)
                    break;
                RandomAccessFile memoryMappedFile = new RandomAccessFile(f, "r");
                int size = (int) f.length();
                MappedByteBuffer out = memoryMappedFile.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, size);
                array = new byte[size];
                out.get(array);
                memoryMappedFile.getChannel().close();
                memoryMappedFile.close();
            }
            list.add(array);
            i++;
        }
        byte[][] output = new byte[list.size()][0];
        return list.toArray(output);
    }

    private String toUpperCaseFirstOne(String s)
    {
        if(Character.isUpperCase(s.charAt(0)))
            return s;
        else
            return (new StringBuilder()).append(Character.toUpperCase(s.charAt(0))).append(s.substring(1)).toString();
    }

    private int skipComment(String str, int start){
        int i = start;
        if(str.charAt(start)=='#'){
            while (i<str.length()){
                char next = str.charAt(i);
                if(next=='\n'){
                    i++;
                    break;
                }
                i++;
            }
        }
        return i;
    }

    private int skipWhiteChar(String str, int start){
        int i = start;
        while(i<str.length()){
            char next = str.charAt(i);
            if(!(next==' '||next=='\t'||next=='\n'||next=='\r'||next=='\f')){//maybe other white char
                break;
            }
            i++;
        }
        return i;
    }

    private int getNumberOffset(String str, int start){
        int i = start;
        while (i<str.length()){
            char next = str.charAt(i);
            if(!((next>='0'&&next<='9')||next=='.'||next=='-')){
                break;
            }
            i++;
        }
        return i;
    }

    private int getKeywordOffset(String str, int start){//if not in quotes, must be [A-Za-z0-9_]
        int i = start;
        if(str.charAt(start)=='"'){
            return str.indexOf('"',start+1) + 1;
        }
        if(str.charAt(start)=='\''){
            return str.indexOf('\'',start+1) + 1;
        }
        while(i<str.length()){
            char next = str.charAt(i);
            if(!((next>='A'&&next<='Z')||(next>='a'&&next<='z')||(next>='0'&&next<='9')||next=='_')){
                break;
            }
            i++;
        }
        return i;
    }

    private int nextToken(String str, int start){
        if(start>=str.length())
            return start;
        if(str.charAt(start)=='{'||str.charAt(start)=='}'||str.charAt(start)==':'){
            return start + 1;
        }
        int offset = getNumberOffset(str,start);
        if(offset!=start)
            return offset;
        offset = getKeywordOffset(str,start);
        return offset;
    }

    private List<String> string2Tokens(String str){
        List<String> tokens = new ArrayList<>();
        int i =0;
        while(i<str.length()){
            i = skipWhiteChar(str, i);
            if(i>=str.length())
                break;
            if(str.charAt(i)=='#') {
                i = skipComment(str, i);
                continue;
            }
            int offset = nextToken(str, i);
            if(offset==i)
                break;
            String token = str.substring(i, offset);
            token = token.replace("\"","").trim().replace("\n","").replace("'","");
            tokens.add(token);
            i = offset;
        }
        return tokens;
    }

    private void addToMap(LinkedHashMap<String, Object> keyValue, String key, Object value)
    {
        Object orgValue = keyValue.get(key);
        if(orgValue==null) {
            keyValue.put(key, value);
        }
        else if(orgValue instanceof List){
            List<Object> list = (List)orgValue;
            list.add(value);
        }
        else {
            List<Object> list = new ArrayList<>();
            list.add(orgValue);
            list.add(value);
            keyValue.put(key, list);
        }
    }

    private int tokensToKeyValues(String[] token, int offset, LinkedHashMap<String,Object> keyValue)
    {
        int i = offset;
        int tmp;
        while(i<token.length){

            if(token[i].equals("}")){
                return i;
            }

            String key = token[i].trim().replace("\"","").replace("\'","");

            if(token[i+1].equals("{") ||
                    (token[i+1].equals(":") && token[i+2].equals("{"))){
                if(token[i+1].equals(":")){
                    i++;
                }
                LinkedHashMap<String, Object> map = new LinkedHashMap<>();
                tmp = tokensToKeyValues(token, i+2, map);
                if(tmp==-1){
                    return -1;
                }
                addToMap(keyValue, key, map);
                i = tmp + 1;//skip the last "}"
                continue;
            }

            if(token[i+1].equals(":")) {
                String value = token[i + 2];
                addToMap(keyValue, key, value);
                i += 3;
                continue;
            }

            LogUtil.e(TAG,"syntax error at (" + i + ")" + token[i]);
            return -1;
        }
        return i;
    }

    private Object parseStringToClass(String paraStr, Class cls)
    {
        Object parsedPara = null;
        try {
            if (cls.equals(String.class)) {
                parsedPara = paraStr;
            } else if (cls.equals(float.class)) {
                parsedPara = Float.parseFloat(paraStr);
            } else if (cls.equals(double.class)) {
                parsedPara = Double.parseDouble(paraStr);
            } else if (cls.equals(int.class)) {
                parsedPara = Integer.parseInt(paraStr);
            } else if (cls.equals(boolean.class)) {
                parsedPara = Boolean.parseBoolean(paraStr);
            }
        }
        catch (NumberFormatException e){
            e.printStackTrace();
            return null;
        }
        return parsedPara;
    }

    private boolean tryInvokeMethod(Layer layer, String methodStr, Object param){
        List<String> paraStr = new ArrayList<>();
        if(param instanceof List){
            List obj = (List) param;
            for(Object o:obj){
                paraStr.add(o.toString());
            }
        }
        else{
            paraStr.add(param.toString());
        }

        try {
            Class paraType = null;
            Object para;
            Method method = null;
            Method[] methods = layer.getClass().getMethods();
            for(Method m:methods){
                if(m.getName().equals(methodStr)){
                    method = m;
                    break;
                }
            }
            if(method==null){
                Log.e(TAG, "no such method "+ methodStr);
                return false;
            }
            Class[] parameters = method.getParameterTypes();
            if(parameters.length!=1){
                Log.e(TAG, "method " + methodStr + " parameters!=1");
                return false;
            }
            Class parameter = parameters[0];
            if (parameter.isArray()){
                paraType = parameter.getComponentType();
                Object paras = Array.newInstance(paraType, paraStr.size());
                for(int i=0;i<paraStr.size();i++){
                    Object value = parseStringToClass(paraStr.get(i),paraType);
                    if(value==null){
                        Log.e(TAG, "can not parse "+paraStr.get(i)+" to "+paraType.toString());
                        return false;
                    }
                    Array.set(paras,i,value);
                }
                para = paras;
            }
            else{
                if(paraStr.size()!=1){
                    Log.e(TAG, "parameter is not array. But there is more than one.");
                    return false;
                }
                para = parseStringToClass(paraStr.get(0),parameter);
                if(para==null){
                    Log.e(TAG, "can not parse "+paraStr.get(0)+" to "+parameter.toString());
                    return false;
                }
            }
            method.invoke(layer,para);
        } catch (InvocationTargetException|IllegalAccessException e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    private String bottomBarToCamel(String str){
        int bar = str.indexOf('_');
        if(bar==-1||bar==str.length()-1)
            return str;
        String toReplace = str.substring(bar,bar+2);
        String upper = str.substring(bar+1,bar+2).toUpperCase();
        return bottomBarToCamel(str.replace(toReplace,upper));
    }

    private Layer loadLayerClass(String name)
    {
        Layer layer = null;
        try {
            layer = (Layer) Class.forName(LAYER_CLASS_PACKAGE + "." + name).newInstance();
        } catch (ClassNotFoundException|IllegalAccessException|InstantiationException e) {
            LogUtil.e(TAG,"class not found:layer <" + name + "> can not be initiated!");
            e.printStackTrace();
        }
        return layer;
    }

    private void setLayerParams(Layer layer, Map<String, Object> paramMap){
        String className = layer.getClass().toString();
        set:for(String para:paramMap.keySet()){
            Object params = paramMap.get(para);
            for(String f:LAYER_PARAMETER_FITER){
                if(f.equals(para)){
                    continue set;
                }
            }
            String dataName = bottomBarToCamel(para);
            String methodName = "set" + toUpperCaseFirstOne(dataName);
            String fullName = className+":"+methodName;
            if(failedMethod.contains(fullName)){
                continue;
            }
            boolean result = tryInvokeMethod(layer, methodName, params);
            if(!result){
                failedMethod.add(fullName);
                LogUtil.w(TAG, "can not invoke method "+fullName);
            }
        }
    }

    private Map<String, Object> parseParams(LinkedHashMap<String, Object> param){
        Map<String, Object> map = new HashMap<>();
        for(String key: param.keySet()) {
            if (key.contains("param")) {
                Object value = param.get(key);
                if(value instanceof LinkedHashMap) {
                    map.putAll(parseSubParams((LinkedHashMap<String, Object>) value, ""));
                }
                else if(value instanceof List){
                    List vl = (List)value;
                    for(Object v:vl){
                        map.putAll(parseSubParams((LinkedHashMap<String, Object>)v, ""));
                    }
                }
            }
        }
        return map;
    }

    private Map<String, Object> parseSubParams(LinkedHashMap<String, Object> param, String prefix){
        Map<String, Object> map = new HashMap<>();
        for(String key: param.keySet()){
            Object value = param.get(key);
            if(value instanceof LinkedHashMap){
                map.putAll(parseSubParams((LinkedHashMap) value, prefix+key+"_"));
            }
            else {
                map.put(prefix + key, value);
            }
        }
        return map;
    }

    private void parseLayer(LinkedHashMap<String,Object> param, LayerGraph graph) throws IOException {
        String layerName = (String)param.get("name");
        String layerType = (String)param.get("type");
        if(param.get("include")!=null){
            LinkedHashMap<String,Object> include = (LinkedHashMap)param.get("include");
            String phase = (String)include.get("phase");
            if(phase.equalsIgnoreCase("TRAIN")){
                graph.setCurrentLayer(null);
                return;
            }
        }

        if("python".equalsIgnoreCase(layerType)){
            layerType = toUpperCaseFirstOne(layerName);
        }

        Layer layer = loadLayerClass(layerType);
        if(layer==null){
            throw new IOException("Can not load Layer "+layerType);
        }

        layer.setRenderScript(renderScript);
        layer.setName(layerName);
        graph.setName(layerName);
        graph.setType(layerType);
        graph.setCurrentLayer(layer);

        Object top = param.get("top");
        Object bottom = param.get("bottom");
        if(top instanceof List) {
            graph.getNextLayerNames().addAll((List)top);
        }
        else if(top instanceof String){
            graph.getNextLayerNames().add((String)top);
        }
        if(bottom instanceof List) {
            graph.getPrevLayerNames().addAll((List)bottom);
        }
        else if(bottom instanceof String){
            graph.getPrevLayerNames().add((String)bottom);
        }

        Map<String, Object> paramMap = parseParams(param);
        setLayerParams(layer, paramMap);

        if(layer instanceof LayerParamInterface){
            LayerParamInterface lpi = (LayerParamInterface) layer;
            byte[][] para = getLayerParameterFromFile(layerName);
            lpi.loadParams(para);
        }
    }

    private void loadInputLayer(LayerGraph graph, LayerGraph start, String name, List<Object> dim) throws IOException {

        graph.setName(name);
        graph.getPrevLayerNames().add(start.getName());
        graph.getNextLayerNames().add(name);
        name = bottomBarToCamel(name);
        name = toUpperCaseFirstOne(name);
        Layer layer = loadLayerClass(name);
        if(layer==null){
            throw new IOException("Can not load Layer "+name);
        }
        layer.setRenderScript(renderScript);
        graph.setType(name);
        graph.setCurrentLayer(layer);
        boolean ret = tryInvokeMethod(layer, "setDim", dim);
        if(!ret){
            LogUtil.w(TAG, "invoke method "+name+":"+"setDim failed");
        }
    }

    private List<LayerGraph> parseGraphs(LinkedHashMap<String, Object> map) throws IOException {
        List<Object> layers = (List<Object>)map.get("layer");
        List<LayerGraph> layerList = new ArrayList<>();

        LayerGraph input = new LayerGraph();
        Layer inputLayer = new Input();
        input.setCurrentLayer(inputLayer);
        input.setName("start");
        input.getNextLayerNames().add("start");
        input.setType("Input");
        layerList.add(input);

        Object inputObject = map.get("input");
        if(inputObject!=null){
            List<String> inputs;
            List<LinkedHashMap> inputShapes;
            if(inputObject instanceof List){
                inputs = (List<String>)inputObject;
                inputShapes = (List)map.get("input_shape");
            }
            else{
                inputs = new ArrayList<>();
                inputs.add((String)inputObject);
                inputShapes = new ArrayList<>();
                inputShapes.add((LinkedHashMap)map.get("input_shape"));
            }

            for(int i=0;i<inputs.size();i++)
            {
                String inputName = inputs.get(i);
                LinkedHashMap inputShapeMap = inputShapes.get(i);
                List inputShape = (List)inputShapeMap.get("dim");
                LayerGraph inputGraph = new LayerGraph();
                loadInputLayer(inputGraph, input, inputName, inputShape);
                layerList.add(inputGraph);
                if (inputGraph.getCurrentLayer() instanceof Data) {
                    Data dataLayer = (Data)inputGraph.getCurrentLayer();
                    ((Input)inputLayer).setDim(dataLayer.getDim());
                }
            }
        }

        for(Object l:layers){
            LinkedHashMap<String, Object> layerParam = (LinkedHashMap)l;
            LayerGraph layer = new LayerGraph();
            parseLayer(layerParam, layer);
            if(layer.getCurrentLayer()!=null) {
                layerList.add(layer);
            }
        }
        return layerList;
    }

    private boolean isSelfConnected(LayerGraph layer)
    {
        List<String> next = layer.getNextLayerNames();
        List<String> prev = layer.getPrevLayerNames();
        return (next.size()==1 && prev.size()==1 && next.get(0).equals(prev.get(0)));
    }

    private List<LayerGraph> getAllLayerByBottomName(List<LayerGraph> list, int index, String bottom)
    {
        List<LayerGraph> out = new ArrayList<>();
        boolean connected = false;

        found:for(int i=index;i<list.size();i++){
            LayerGraph layer = list.get(i);
            boolean selfConnected = isSelfConnected(layer);
            for(String bt:layer.getPrevLayerNames()){
                if(bt.equals(bottom)){
                    if(selfConnected){
                        if(!connected){
                            out.add(layer);
                        }
                        break found;
                    }else{
                        out.add(layer);
                        connected = true;
                    }
                }
            }
        }
        return out;
    }

    private void connectLayerGraph(List<LayerGraph> list){
        for(LayerGraph l:list){
            l.initInputData();
        }
        for(int i=0;i<list.size();i++){
            LayerGraph layer = list.get(i);
            String top = layer.getNextLayerNames().get(0);
            List<LayerGraph> next = getAllLayerByBottomName(list, i+1, top);
            if(next.isEmpty()){
                LogUtil.i(TAG,"the last layer is:"+top+".");
                return;
            }
            layer.getNextLayer().addAll(next);
            layer.setOutPutName(top);
            if(isSelfConnected(layer)){
                layer.getCurrentLayer().setSelfConnected(true);
            }
        }
        for(int i=0;i<list.size();i++){
            LayerGraph layer = list.get(i);
            if(layer.getPrevLayerNames().size()!=0){
                layer.getPrevLayerNames().clear();
            }
        }
        for(int i=0;i<list.size();i++){
            for(LayerGraph next:list.get(i).getNextLayer()){
                next.getPrevLayerNames().add(list.get(i).getName());
            }
        }
    }

    private StringBuffer elementToString(Object value, String prefix){
        StringBuffer sb = new StringBuffer();
        if(value instanceof LinkedHashMap){
            sb.append(jsonToString((LinkedHashMap)value, prefix));
        }
        else if(value instanceof List){
            sb.append(listToString((List)value, prefix));
        }
        else if(value instanceof String){
            sb.append("\"");
            sb.append(value);
            sb.append("\"");
        }
        else {
            sb.append(value);
        }
        return sb;
    }

    private StringBuffer listToString(List obj, String prefix){
        StringBuffer sb = new StringBuffer();
        sb.append("[");
        sb.append("\n");
        String newPrefix = prefix + "  ";
        int length = obj.size();
        for(Object o:obj){
            sb.append(newPrefix);
            sb.append(elementToString(o, newPrefix));
            length--;
            if(length>0){
                sb.append(",");
            }
            sb.append("\n");
        }
        sb.append(prefix);
        sb.append("]");
        return sb;
    }

    private StringBuffer jsonToString(LinkedHashMap<String, Object> map, String prefix){
        StringBuffer sb = new StringBuffer();
        sb.append("{");
        sb.append("\n");
        String newPrefix = prefix + "  ";
        int length = map.keySet().size();
        for(String s:map.keySet()){
            sb.append(newPrefix);
            sb.append("\"");
            sb.append(s);
            sb.append("\" ");
            sb.append(": ");
            Object value = map.get(s);
            sb.append(elementToString(value, newPrefix));
            length--;
            if(length>0){
                sb.append(",");
            }
            sb.append("\n");
        }
        sb.append(prefix);
        sb.append("}");
        return sb;
    }

    private void saveToJson(LinkedHashMap<String, Object> map, String path){
        String json = new String(jsonToString(map,""));
        LogUtil.w(TAG,json);
        try {
            FileOutputStream fos = new FileOutputStream(path);
            fos.write(json.getBytes());
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private LayerGraph parse(InputStream protoFile) throws IOException {
        byte[] buff = new byte[protoFile.available()];
        int len = protoFile.read(buff);
        if(len!=buff.length){
            return null;
        }
        String str = new String(buff);
        List<String> tokens = string2Tokens(str);
        LinkedHashMap<String, Object> map = new LinkedHashMap<>();
        int ret = tokensToKeyValues(tokens.toArray(new String[tokens.size()]), 0, map);
        if(ret==-1){
            return null;
        }
        List<LayerGraph> layers = parseGraphs(map);
        connectLayerGraph(layers);
        if(layers.size()==0){
            return null;
        }
        LayerGraph graph = layers.get(0);
        graph.setName((String)map.get("name"));
        return graph;
    }

    public void init(LayerGraph graph){
        graph.init();
    }


    public LayerGraph parseFromRawDataOnAsset(String protoFile, AssetManager asset, String dataDir) throws IOException {
        baseDir = dataDir;
        this.asset = asset;
        useAsset = true;
        InputStream is = asset.open(protoFile);

        LayerGraph layer = parse(is);
        is.close();
        return layer;
    }
    public LayerGraph parseFromRawDataOnStorage(String protoFile, String dataDir) throws IOException {
        baseDir = dataDir;
        LogUtil.i(TAG, "proto file is " + protoFile);
        InputStream is = new FileInputStream(protoFile);
        LayerGraph layer = parse(is);
        is.close();
        return layer;
    }
}
