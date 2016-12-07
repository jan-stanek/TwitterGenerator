package cz.fit.pdd.TwitterPredict;

import com.google.common.base.Charsets;
import com.google.common.io.Files;

import javax.annotation.PostConstruct;
import javax.enterprise.context.ApplicationScoped;
import javax.inject.Named;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

@Named
@ApplicationScoped
public class LstmBean {
    public static Map<Character, Integer> charToIndexMap;
    public static Character[] indexToCharArray;
    public static Set<Character> characterSet;

    private Lstm lstm;

    @PostConstruct
    public void init() {
        try {
            File dataFile = new File("data.txt");
            String data = Files.toString(dataFile, Charsets.UTF_8);

            createCharacterSet(data);

            File netFile = new File("net.zip");
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
    }

    public String getGeneratedTweet() {
        return lstm.generate();
    }
}
