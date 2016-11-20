package twitter;

import com.google.common.base.CharMatcher;
import org.apache.commons.lang3.StringUtils;
import twitter4j.*;
import twitter4j.conf.ConfigurationBuilder;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jan on 13.11.2016.
 */
public class TwitterFacade {

    private Twitter twitter;

    public TwitterFacade() {

        ConfigurationBuilder cb = new ConfigurationBuilder();
        cb.setDebugEnabled(true)
                .setOAuthConsumerKey("mWyX6Rhqd7fDLfg6kb33r252e")
                .setOAuthConsumerSecret("BXl8cYsmmqynbarBQ9zf4khR1tRHHZrDyPMRuZw5Lv1R8rx0sq")
                .setOAuthAccessToken("716625760-qCuakk81wmQvMdXikYngpSWBiL7ubXEkDwzS5rOj")
                .setOAuthAccessTokenSecret("fWaab06ElOPGtcKsNZfyUOrFm1538knK2nclx9osaVkGu");

        TwitterFactory twitterFactory = new TwitterFactory(cb.build());
        twitter = twitterFactory.getInstance();
    }

    public List<String> getTweets(String user) throws TwitterException {
        int page = 1;

        List<Status> statuses = null;
        List<String> tweets = new ArrayList<>();

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

                tweetText = StringUtils.stripAccents(tweetText);

                tweets.add(tweetText);
            }
        }

        return tweets;
    }
}
