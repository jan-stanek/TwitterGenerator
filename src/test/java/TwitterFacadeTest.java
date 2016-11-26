import org.junit.Before;
import org.junit.Test;
import twitter.TwitterFacade;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Properties;

/**
 * Created by jan on 13.11.2016.
 */
public class TwitterFacadeTest {

    TwitterFacade twitterFacade;

    @Before
    public void setUp() throws Exception {
        Properties config = new Properties();
        InputStream configFile = new FileInputStream("config.properties");
        config.load(configFile);
        twitterFacade = new TwitterFacade(config);
    }

    @Test
    public void getTweets() throws Exception {
        String[] arr = {""};
        twitterFacade.getTweets(arr);
    }

}