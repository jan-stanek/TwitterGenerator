package cz.fit.pdd.TwitterPredict;

import javax.annotation.PostConstruct;
import javax.enterprise.context.RequestScoped;
import javax.inject.Inject;
import javax.inject.Named;

@Named
@RequestScoped
public class TweetBean {

    private String tweet;

    @Inject
    private LstmBean lstmBean;

    @PostConstruct
    void init() {
        tweet = lstmBean.getGeneratedTweet();
    }

    public String getTweet() {
        return tweet;
    }
}
