package cz.fit.pdd.TwitterPredict;

import com.google.common.base.Charsets;
import com.google.common.io.Files;

import javax.annotation.PostConstruct;
import javax.enterprise.context.ApplicationScoped;
import javax.inject.Named;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.net.URL;
import java.net.URLDecoder;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

@Named
@ApplicationScoped
public class LstmBean implements Serializable {
    public static Map<Character, Integer> charToIndexMap;
    public static Character[] indexToCharArray;
    public static Set<Character> characterSet;
    public static Set<Character> startCharactersSet;

    private Lstm lstm;

    @PostConstruct
    public void init() {
        try {
            ClassLoader classloader = Thread.currentThread().getContextClassLoader();

            String dataPath = classloader.getResource("tmp/data.txt").getFile();
            URL dataUrl = new URL("file://" + dataPath);
            File dataFile = new File(URLDecoder.decode(dataUrl.getFile(), "UTF-8" ));
            String data = Files.toString(dataFile, Charsets.UTF_8);
            createCharacterSet(data);

            String netPath = classloader.getResource("tmp/net.zip").getFile();
            URL netUrl = new URL("file://" + netPath);
            File netFile = new File(URLDecoder.decode(netUrl.getFile(), "UTF-8" ));
            lstm = new Lstm(netFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void createCharacterSet(String data) {
        characterSet = new HashSet<>();

        for (char c : data.toCharArray()) {
            characterSet.add(c);
        }

        charToIndexMap = new HashMap<>();
        indexToCharArray = new Character[characterSet.size()];

        int i = 0;
        for (Character c : characterSet) {
            charToIndexMap.put(c, i);
            indexToCharArray[i] = c;
            i++;
        }

        startCharactersSet = new HashSet<>();

        String[] lines = data.split("\0");
        for (String s : lines) {
            startCharactersSet.add(s.charAt(0));
        }
    }

    public String getGeneratedTweet() {
        return lstm.generate();
    }
}
