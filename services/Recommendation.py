import pickle

class RecommendationService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            
            cls._instance = super(RecommendationService, cls).__new__(cls)
            cls._instance.load_recommendation_model()
            print("Initialized Singleton")
        return cls._instance
    
    def __call__(self):
        return self

    def load_recommendation_model(self):
        self.model_data = None
        self.svd = None
        self.user_item = None
        self.collab_sim = None
        try:
            with open("recommendation_model.pkl", "rb") as f:
                model_data = pickle.load(f)
                print(model_data)
                self.model_data = model_data
                self.svd = model_data["svd"]
                self.user_item = model_data["user_item"]
                self.collab_sim = model_data["collab_sim"]
        except Exception as e:
            print(f"Error loading recommendation model: {e}")

    def top_n_recommendations(self, customer_id: str, num_recommendations: int = 3):
            if self.user_item is None:
                raise RuntimeError("Recommendation model is not loaded.")
            if customer_id not in self.user_item.index:
                return []
            idx = list(self.user_item.index).index(customer_id)

            similar_users = self.collab_sim[idx].argsort()[::-1][1:6]
            scores = self.user_item.iloc[similar_users].mean()

            user_services = self.user_item.loc[customer_id]
            # include for recommendations for services not tried by users
            scores = scores[user_services == 0] 
            return list(scores.sort_values(ascending=False).head(num_recommendations).index)


def get_recommendation_service():
    recommendation_service = RecommendationService()
    yield recommendation_service
    