data {

        

        int NT; // this gives the MAX len of trials

        int NS; 

        int NT_all[NS]; // a vector with different trial lengths per subject


        int r[NS,NT];

        int c[NS,NT];



}


parameters {

        // normal intercept model
        real betam;
        real alpham;
        real staym;


        real<lower=0> betasd;
        real<lower=0> alphasd;
        real<lower=0> staysd;



        real betas[NS];
        real alphas[NS];
        real stay[NS];

}


model {

        betam ~ normal(0,2);

        alpham ~ normal(0,2);

        betas ~ normal(0,2);

        alphas ~ normal(0,2);




        for (s in 1:NS) {

                real alpha;

                real q[2];

                real pc; // past choice


                betas[s] ~ normal(betam ,betasd);

                alphas[s] ~ normal(alpham,alphasd);

                stay[s] ~ normal(staym, staysd);

                alpha <- Phi_approx(alphas[s]);


                for (i in 1:2) {

                        q[i] <- 0; 

                }


                for (t in 1:NT_all[s]) {

                        if (t > 1) {
                                pc = c[s, (t - 1)] * 2 - 1;
                        } else {
                                pc = 0;
                        }

                        c[s,t] ~ bernoulli_logit(betas[s]  * (q[2] - q[1]) + stay[s] * pc);


                        q[c[s,t]+1] <- (1-alpha) * q[c[s,t]+1] + alpha * r[s,t];

                }

        }

}



    

