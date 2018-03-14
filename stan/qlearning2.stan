data {

        int NT; // this gives the MAX len of trials

        int NS; 

        int NT_all[NS]; // a vector with different trial lengths per subject


        int r[NS,NT];

        int c[NS,NT];



}


parameters {

        real betam;
        real alpham;


        real<lower=0> betasd;
        real<lower=0> alphasd;



        real betas[NS];
        real alphas[NS];

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

                alpha <- Phi_approx(alphas[s]);


                for (i in 1:2) {

                        q[i] <- 0; 

                }


                for (t in 1:NT_all[s]) {

                        c[s,t] ~ bernoulli_logit(betas[s]  * (q[2] - q[1]));


                        q[c[s,t]+1] <- (1-alpha) * q[c[s,t]+1] + alpha * r[s,t];

                }

        }

}

/*
// take out generated quantities to speed up. I calculate the quantities per sample in python and not in Stan
generated quantities {

    real delta[NS, NT];
    vector[2] prob_c[NS, NT];
    int c_hat[NS, NT];


    /*
    real q_ipsa[NS, NT + 1];
    real q_contra[NS, NT + 1];
    */

    // initialize
    for (s in 1:NS) {for (t in 1:NT) {delta[s,t] = 0;}}
    for (s in 1:NS) {for (t in 1:NT) {for (i in 1:2) {prob_c[s,t,i] = 0;}}}
    for (s in 1:NS) {for (t in 1:NT) {c_hat[s,t] = -5;}}

    /*
    for (s in 1:NS) {for (t in 1:(NT +1))  {q_ipsa[s,t] = 0;}} 

    for (s in 1:NS) {for (t in 1:(NT +1)) {q_contra[s,t] = 0;}}
    */

    for (s in 1:NS) {
        vector[2] q;
        real alpha;

        alpha = Phi_approx(alphas[s]);
        
        for (i in 1:2) {q[i] <- 0;}

        for (t in 1:NT_all[s]) { 
            delta[s,t] = r[s,t] - q[c[s,t]+1];
            q[c[s,t]+1] = q[c[s,t]+1] * (1 - alpha) + alpha * r[s, t];
            prob_c[s, t] = softmax(betas[s] * q);
            c_hat[s, t] = categorical_logit_rng(softmax(betas[s] * q));
            


        }


    }
}
*/
    

