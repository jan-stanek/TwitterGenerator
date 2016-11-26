package lstm;

import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;


public class StringCleaner {

    private List<String> texts;

    public StringCleaner(List<String> texts) {
        this.texts = texts;
    }

    public StringCleaner removeDiacritic() {
        List<String> tmp = new ArrayList<>();

        for (String tweet : texts) {
            tmp.add(StringUtils.stripAccents(tweet));
        }

        texts = tmp;

        return this;
    }

    public StringCleaner replaceChars(Map<Character, Character> replaceChars) {
        List<String> tmp = new ArrayList<>();

        for (String text : texts) {
            for (int i = 0; i < text.length(); i++) {
                Character c = text.charAt(i);
                if (replaceChars.containsKey(c)) {
                    text = text.replace(c, replaceChars.get(c));
                }
            }
            tmp.add(text);
        }

        texts = tmp;

        return this;
    }

    public StringCleaner removeInvalidChars(Set<Character> validChars) {
        List<String> tmp = new ArrayList<>();

        for (String text : texts) {
            text = text.replaceAll("[^\\x00-\\x7F]", "");

            for (int i = 0; i < text.length(); i++) {
                Character c = text.charAt(i);
                if (!validChars.contains(c)) {
                    text = text.replace(c.toString(), "");
                    int a = 1;
                }
            }

            tmp.add(text);
        }

        texts = tmp;

        return this;
    }

    public StringCleaner addNewLines() {
        List<String> tmp = new ArrayList<>();

        for (String text : texts) {
            text = text.concat("\n");
            tmp.add(text);
        }

        texts = tmp;

        return this;
    }

    public List<String> getData() {
        return texts;
    }
}
