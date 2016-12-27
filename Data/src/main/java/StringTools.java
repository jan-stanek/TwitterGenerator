import org.apache.commons.lang3.StringUtils;

import java.util.*;


public class StringTools {

    private List<String> texts;

    public StringTools(List<String> texts) {
        this.texts = texts;
    }

    public StringTools removeDiacritic() {
        List<String> tmp = new ArrayList<>();

        for (String tweet : texts) {
            tmp.add(StringUtils.stripAccents(tweet));
        }

        texts = tmp;

        return this;
    }

    public StringTools replaceChars(Map<Character, Character> replaceChars) {
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

    public StringTools toLowerCase() {
        List<String> tmp = new ArrayList<>();

        for (String text : texts) {
            tmp.add(text.toLowerCase());
        }

        texts = tmp;

        return this;
    }

    public StringTools copy(int count) {
        List<String> tmp = new ArrayList<>();

        for (int i = 0; i < count; i++) {
            for (String text : texts) {
                tmp.add(text.toLowerCase());
            }
        }

        texts = tmp;

        return this;
    }

    public StringTools removeInvalidChars(Set<Character> validChars) {
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

    public StringTools removeNoLetterTexts() {
        List<String> tmp = new ArrayList<>();

        Set<Character> letters = new HashSet<>();
        for (char c = 'a'; c <= 'z'; c++)
            letters.add(c);
        for (char c = 'A'; c <= 'Z'; c++)
            letters.add(c);

        for (String text : texts) {
            for (char c : text.toCharArray()) {
                if (letters.contains(c)) {
                    tmp.add(text);
                    break;
                }
            }
        }

        texts = tmp;

        return this;
    }

    public StringTools removeMultipleSpaces() {
        List<String> tmp = new ArrayList<>();

        Set<Character> letters = new HashSet<>();
        for (char c = 'a'; c <= 'z'; c++)
            letters.add(c);
        for (char c = 'A'; c <= 'Z'; c++)
            letters.add(c);

        for (String text : texts) {
            tmp.add(text.replaceAll("^ +| +$|( )+", "$1"));
        }

        texts = tmp;

        return this;
    }

    public StringTools addNewLines() {
        List<String> tmp = new ArrayList<>();

        for (String text : texts) {
            text = text.concat("\0");
            tmp.add(text);
        }

        texts = tmp;

        return this;
    }

    public List<String> getListData() {
        return texts;
    }

    public String getStringData() {
        String res = "";
        for(String text : texts) {
            res = res.concat(text);
        }
        return res;
    }
}
