aspects=["DRINKS", "DRINKS_ALCOHOL", "DRINKS_ALCOHOL_BEER", "DRINKS_ALCOHOL_HARD", "DRINKS_ALCOHOL_LIGHT", "DRINKS_ALCOHOL_WINE", "DRINKS_NON-ALCOHOL_COLD", "DRINKS_NON-ALCOHOL_HOT", "EXPERIENCE", "EXPERIENCE_BONUS", "EXPERIENCE_COMPANY", "EXPERIENCE_OCCASION", "EXPERIENCE_RECOMMENDATIONS", "EXPERIENCE_RESERVATION", "EXPERIENCE_TAKEOUT", "EXPERIENCE_TIME", "FOOD_FOOD", "FOOD_FOOD_BREAD", "FOOD_FOOD_CHEESE", "FOOD_FOOD_CHICKEN", "FOOD_FOOD_DESSERT", "FOOD_FOOD_DISH", "FOOD_FOOD_EGGS", "FOOD_FOOD_FRUIT", "FOOD_FOOD_MEAT", "FOOD_FOOD_MEAT_BACON", "FOOD_FOOD_MEAT_BEEF", "FOOD_FOOD_MEAT_BURGER", "FOOD_FOOD_MEAT_LAMB", "FOOD_FOOD_MEAT_PORK", "FOOD_FOOD_MEAT_RIB", "FOOD_FOOD_MEAT_STEAK", "FOOD_FOOD_MEAT_VEAL", "FOOD_FOOD_SALAD", "FOOD_FOOD_SAUCE", "FOOD_FOOD_SEAFOOD", "FOOD_FOOD_SEAFOOD_FISH", "FOOD_FOOD_SEAFOOD_SEA", "FOOD_FOOD_SIDE", "FOOD_FOOD_SIDE_PASTA", "FOOD_FOOD_SIDE_POTATO", "FOOD_FOOD_SIDE_RICE", "FOOD_FOOD_SIDE_VEGETABLES", "FOOD_FOOD_SOUP", "FOOD_FOOD_SUSHI", "FOOD_MEALTYPE_BREAKFAST", "FOOD_MEALTYPE_BRUNCH", "FOOD_MEALTYPE_DINNER", "FOOD_MEALTYPE_LUNCH", "FOOD_MEALTYPE_MAIN", "FOOD_MEALTYPE_START", "FOOD_PORTION", "FOOD_SELECTION", "GENERAL", "PERSONAL", "RESTAURANT", "RESTAURANT_ATMOSPHERE", "RESTAURANT_CUSINE", "RESTAURANT_ENTERTAINMENT_MUSIC", "RESTAURANT_ENTERTAINMENT_SPORT", "RESTAURANT_INTERIOR", "RESTAURANT_INTERNET", "RESTAURANT_LOCATION", "RESTAURANT_MONEY", "RESTAURANT_PARKING", "SERVICE"]
category=[]
liquids=[]
liquids={"DRINKS", "DRINKS_ALCOHOL", "DRINKS_ALCOHOL_BEER", "DRINKS_ALCOHOL_HARD", "DRINKS_ALCOHOL_LIGHT", "DRINKS_ALCOHOL_WINE", "DRINKS_NON-ALCOHOL_COLD", "DRINKS_NON-ALCOHOL_HOT"}
food_variety=[]
food_variety={"FOOD_FOOD", "FOOD_FOOD_BREAD", "FOOD_FOOD_CHEESE", "FOOD_FOOD_CHICKEN", "FOOD_FOOD_DESSERT", "FOOD_FOOD_DISH", "FOOD_FOOD_EGGS", "FOOD_FOOD_FRUIT", "FOOD_FOOD_MEAT", "FOOD_FOOD_MEAT_BACON", "FOOD_FOOD_MEAT_BEEF", "FOOD_FOOD_MEAT_BURGER", "FOOD_FOOD_MEAT_LAMB", "FOOD_FOOD_MEAT_PORK", "FOOD_FOOD_MEAT_RIB", "FOOD_FOOD_MEAT_STEAK", "FOOD_FOOD_MEAT_VEAL", "FOOD_FOOD_SALAD", "FOOD_FOOD_SAUCE", "FOOD_FOOD_SEAFOOD", "FOOD_FOOD_SEAFOOD_FISH", "FOOD_FOOD_SEAFOOD_SEA", "FOOD_FOOD_SIDE", "FOOD_FOOD_SIDE_PASTA", "FOOD_FOOD_SIDE_POTATO", "FOOD_FOOD_SIDE_RICE", "FOOD_FOOD_SIDE_VEGETABLES", "FOOD_FOOD_SOUP", "FOOD_FOOD_SUSHI"}
food_timing={}
food_timing={"FOOD_MEALTYPE_BREAKFAST", "FOOD_MEALTYPE_BRUNCH", "FOOD_MEALTYPE_DINNER", "FOOD_MEALTYPE_LUNCH", "FOOD_MEALTYPE_MAIN", "FOOD_MEALTYPE_START", "FOOD_PORTION", "FOOD_SELECTION"}
restaurant_features=[]
restaurant_features={"GENERAL", "PERSONAL", "RESTAURANT", "RESTAURANT_ATMOSPHERE", "RESTAURANT_CUSINE", "RESTAURANT_ENTERTAINMENT_MUSIC", "RESTAURANT_ENTERTAINMENT_SPORT", "RESTAURANT_INTERIOR", "RESTAURANT_INTERNET", "RESTAURANT_LOCATION", "RESTAURANT_MONEY", "RESTAURANT_PARKING", "SERVICE"}
cat=[{'liquids','food_variety','food_timing','restaurant_features'}]
category={}
category={'liquids':{"DRINKS", "DRINKS_ALCOHOL", "DRINKS_ALCOHOL_BEER", "DRINKS_ALCOHOL_HARD", "DRINKS_ALCOHOL_LIGHT", "DRINKS_ALCOHOL_WINE", "DRINKS_NON-ALCOHOL_COLD", "DRINKS_NON-ALCOHOL_HOT"},'food_variety':{"FOOD_FOOD", "FOOD_FOOD_BREAD", "FOOD_FOOD_CHEESE", "FOOD_FOOD_CHICKEN", "FOOD_FOOD_DESSERT", "FOOD_FOOD_DISH", "FOOD_FOOD_EGGS", "FOOD_FOOD_FRUIT", "FOOD_FOOD_MEAT", "FOOD_FOOD_MEAT_BACON", "FOOD_FOOD_MEAT_BEEF", "FOOD_FOOD_MEAT_BURGER", "FOOD_FOOD_MEAT_LAMB", "FOOD_FOOD_MEAT_PORK", "FOOD_FOOD_MEAT_RIB", "FOOD_FOOD_MEAT_STEAK", "FOOD_FOOD_MEAT_VEAL", "FOOD_FOOD_SALAD", "FOOD_FOOD_SAUCE", "FOOD_FOOD_SEAFOOD", "FOOD_FOOD_SEAFOOD_FISH", "FOOD_FOOD_SEAFOOD_SEA", "FOOD_FOOD_SIDE", "FOOD_FOOD_SIDE_PASTA", "FOOD_FOOD_SIDE_POTATO", "FOOD_FOOD_SIDE_RICE", "FOOD_FOOD_SIDE_VEGETABLES", "FOOD_FOOD_SOUP", "FOOD_FOOD_SUSHI"},'food_timing':{"FOOD_MEALTYPE_BREAKFAST", "FOOD_MEALTYPE_BRUNCH", "FOOD_MEALTYPE_DINNER", "FOOD_MEALTYPE_LUNCH", "FOOD_MEALTYPE_MAIN", "FOOD_MEALTYPE_START", "FOOD_PORTION", "FOOD_SELECTION"},'restaurant_features':{"GENERAL", "PERSONAL", "RESTAURANT", "RESTAURANT_ATMOSPHERE", "RESTAURANT_CUSINE", "RESTAURANT_ENTERTAINMENT_MUSIC", "RESTAURANT_ENTERTAINMENT_SPORT", "RESTAURANT_INTERIOR", "RESTAURANT_INTERNET", "RESTAURANT_LOCATION", "RESTAURANT_MONEY", "RESTAURANT_PARKING", "SERVICE"}}
print("Following are the categories of aspects");
for n,i in enumerate(cat):
    print(i);
#print("Enter the category name");
user_cat=input("Enter the category name : ");
print("Following are the aspects in this category");
print(category[user_cat]);
user_aspect=input("Enter the aspect : ");
z=-1;
for n,i in enumerate(aspects):
    if(user_aspect==i):
        z=n;
print(z);