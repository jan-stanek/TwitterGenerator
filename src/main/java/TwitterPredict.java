import lstm.CharIterator;
import lstm.Lstm;
import twitter.TwitterFacade;
import twitter4j.TwitterException;

import java.io.IOException;
import java.util.List;
import java.util.Random;

public class TwitterPredict {
    public static void main(String[] args) {
        TwitterFacade twitterFacade = new TwitterFacade();

        List<String> tweets;
        CharIterator charIterator;
        Lstm lstm;

        try {
            tweets = twitterFacade.getTweets("PREZIDENTmluvci");
            charIterator = new CharIterator(tweets, 10, 10, new Random());
            lstm = new Lstm(charIterator);
            lstm.train();
        } catch (TwitterException e) {
            e.printStackTrace();
            return;
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
    }
}
