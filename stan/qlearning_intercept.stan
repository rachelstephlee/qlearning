data {

        /

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


        real<lower=0> betasd;
        real<lower=0> alphasd;



        real betas[NS];
        real alphas[NS];
        real beta0[NS];

}


model {

        betam ~ normal(0,2);

        alpham ~ normal(0,2);

        betas ~ normal(0,2);

        alphas ~ normal(0,2);




        for (s in 1:NS) {

                real alpha;

                real q[2];


                betas[s] ~ normal(betam ,betasd);

                alphas[s] ~ normal(alpham,alphasd);

                b0[s] ~ normal(b0m, b0sd)

                alpha <- Phi_approx(alphas[s]);


                for (i in 1:2) {

                        q[i] <- 0; 

                }


                for (t in 1:NT_all[s]) {

                        c[s,t] ~ bernoulli_logit(b0[s]+ betas[s]  * (q[2] - q[1]));


                        q[c[s,t]+1] <- (1-alpha) * q[c[s,t]+1] + alpha * r[s,t];

                }

        }

}



    

