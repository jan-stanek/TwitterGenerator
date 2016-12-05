package twitter;

import com.google.common.base.CharMatcher;
import twitter4j.*;
import twitter4j.conf.ConfigurationBuilder;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;


public class TwitterFacade {

    private Twitter twitter;

    public TwitterFacade(Properties config) {
        ConfigurationBuilder cb = new ConfigurationBuilder();
        cb.setDebugEnabled(true)
                .setOAuthConsumerKey(config.getProperty("consumer_key"))
                .setOAuthConsumerSecret(config.getProperty("consumer_secret"))
                .setOAuthAccessToken(config.getProperty("access_token"))
                .setOAuthAccessTokenSecret(config.getProperty("access_token_secret"));

        TwitterFactory twitterFactory = new TwitterFactory(cb.build());
        twitter = twitterFactory.getInstance();
    }

    public List<String> getTweets(String[] users) throws TwitterException {
        int page;

        List<Status> statuses = null;
        List<String> tweets = new ArrayList<>();

        for (String user : users) {
            page = 1;

            while (true) {
                Paging paging = new Paging(page++, 200);
                statuses = twitter.getUserTimeline(user, paging);

                if (statuses.isEmpty())
                    break;

                for (Status status : statuses) {
                    if (status.isRetweet())
                        continue;

                    String tweetText = status.getText();

                    for (MediaEntity mediaEntity : status.getMediaEntities()) {
                        String url = mediaEntity.getURL();
                        tweetText = tweetText.replace(url, "");
                        tweetText = CharMatcher.anyOf("- :").trimFrom(tweetText);
                    }

                    for (URLEntity urlEntity : status.getURLEntities()) {
                        String url = urlEntity.getText();
                        tweetText = tweetText.replace(url, "");
                        tweetText = CharMatcher.anyOf("- :").trimFrom(tweetText);
                    }

                    if (tweetText.length() == 0)
                        continue;

                    tweets.add(tweetText);
                }
            }
        }

        return tweets;
    }
}
