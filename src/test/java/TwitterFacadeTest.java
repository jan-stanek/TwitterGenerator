import org.junit.Before;
import org.junit.Test;
import twitter.TwitterFacade;

/**
 * Created by jan on 13.11.2016.
 */
public class TwitterFacadeTest {

    TwitterFacade twitterFacade;

    @Before
    public void setUp() throws Exception {
        twitterFacade = new TwitterFacade();
    }

    @Test
    public void getTweets() throws Exception {
        twitterFacade.getTweets("");
    }

}