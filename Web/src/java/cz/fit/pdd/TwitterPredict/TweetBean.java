package cz.fit.pdd.TwitterPredict;

import javax.annotation.PostConstruct;
import javax.enterprise.context.RequestScoped;
import javax.enterprise.context.SessionScoped;
import javax.inject.Inject;
import javax.inject.Named;
import java.io.Serializable;

@Named
@RequestScoped
public class TweetBean implements Serializable {

    private String tweet;

    @Inject
    private LstmBean lstmBean;

    @PostConstruct
    public void init() {

    }

    public String getTweet() {
        return lstmBean.getGeneratedTweet();
    }
}
