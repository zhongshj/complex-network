In this part, we have simulated the viral spreading procedure for this network. Furthermore, we have implemented possible immunity strategy to control the spreading in the procedure. Our intention is to figure out relationship between characteristics of immunized user and immunity performance, then summarize effective strategies for such immunities. 



### Viral spreading rules

1. An email sent from an infected user will surely infect the receiver, but the receiver can only infect other users in the next day. 
2. The virus will be planted to a random user only once in a random day. 
3. 100 user will be immunized in the same day of virus planted. An immunized user will not be infected. 
4. We assume the temporal network after immunity action is unknown, which means, the immunity strategy can only be based on former user behavior. 



### Immunity strategy

Based on these assumptions, we have started our simulation. As for which nodes to be immunized, we have came up with these strategies: 

1. Users registered early.
2. Users that have contacted most other users(in degree and out degree). 
3. Users that have largest mailing frequency. 
4. Users that have largest active days(count days that have sending or receiving activity)



### Spreading simulation 

We have selected the starting day at 25 and 125 respectively to visualize the average infect number while spreading. These two starting days are selected by regarding the whole temporal network as two periods: increasing period where user increased fast and with frequent activity, and stable period where user number remains stable and with less frequent activity. 



(两张图)



From the two figures we can see the spreading gets slow as time goes. However, the spreading speed at the start period is significantly fast. As for immune strategy, immunizing early users is the most effecive way of preventing infections. 



### Immunity performance



(四张图)



These figures shows the immunity performance by different starting day. From these figures we can see that the best strategy for immunity is to block early users. However, this may result from the specific characteristics that the early users have been more active than other all the time(can be seen in the first figure). Therefore, when we leave out the early user strategy(to make it more general), none of the other strategies significantly outperform others though they are all effective in some extent. 



