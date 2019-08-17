library("tidyverse")
library('car')

# cleaning df
ak_total <- read_csv('clean_ak_df')
ak_small <- drop_na(ak_total)

ak_small <- ak_small[-c(48, 49, 119, 116, 10), ]

#### working with the clean df, therefor we have all variables

lm.fit <- lm(price ~ .-location - index - defect - defect_code, data = ak_small)
summary(lm.fit)

vif(lm.fit) # not too much collinearity

par(mfrow=c(2,2))
plot(lm.fit) # residuals look very messed up 


lm.fit_sig <- lm(price ~ .-location - index - defect - defect_code - carbon - aluminium - niobium - weight - `linear feat`, data = ak_small)
summary(lm.fit_sig) # r squared does not change at all after removing nosig variables
vif(lm.fit_sig) # no collineratiy
plot(lm.fit_sig)
plot(hatvalues(lm.fit_sig))

# trying to remove 48 and 49 observations

which.max(hatvalues(lm.fit_sig))

anova(lm.fit_sig, lm.fit)

summary(lm.fit_sig)

####


