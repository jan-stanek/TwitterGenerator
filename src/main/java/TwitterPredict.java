import lstm.CharIterator;
import lstm.Lstm;
import lstm.StringCleaner;
import twitter.TwitterFacade;
import twitter4j.TwitterException;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;

public class TwitterPredict {

    public static void main(String[] args) {
        Properties config = new Properties();

        List<String> tweets;
        Lstm net;

        Map<Character, Character> replaceChars = generateReplaceChars();
        Set<Character> validChars = generateValidChars();

        try {
            InputStream configFile = new FileInputStream("config.properties");
            config.load(configFile);

            TwitterFacade twitterFacade = new TwitterFacade(config);

            tweets = twitterFacade.getTweets(args);
            tweets = new StringCleaner(tweets)
                    .removeDiacritic()
                    .replaceChars(replaceChars)
                    .removeInvalidChars(validChars)
                    .addNewLines()
                    .getData();

            net = new Lstm(validChars);
            net.train(tweets);
            net.generate();
        } catch (TwitterException | IOException e) {
            e.printStackTrace();
            return;
        }
    }

    private static Set<Character> generateValidChars() {
        Set<Character> validChars = new HashSet<>();

        for (char c = '0'; c <= '9'; c++)
            validChars.add(c);

        for (char c = 'a'; c <= 'z'; c++)
            validChars.add(c);

        for (char c = 'A'; c <= 'Z'; c++)
            validChars.add(c);

        validChars.add('\n');
        validChars.add(' ');
        validChars.add('!');
        validChars.add('"');
        validChars.add('#');
        validChars.add('%');
        validChars.add('&');
        validChars.add('(');
        validChars.add(')');
        validChars.add('+');
        validChars.add('-');
        validChars.add(',');
        validChars.add('.');
        validChars.add('/');
        validChars.add(':');
        validChars.add('=');
        validChars.add('?');
        validChars.add('@');

        return validChars;
    }

    private static Map<Character, Character> generateReplaceChars() {
        Map<Character, Character> replaceChars = new HashMap<>();

        replaceChars.put('“', '"');
        replaceChars.put('”', '"');
        replaceChars.put('„', '"');
        replaceChars.put('\n', ' ');
        replaceChars.put('\\', ' ');

        return replaceChars;
    }
}
