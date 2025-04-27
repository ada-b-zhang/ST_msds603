from metaflow import FlowSpec, step

class Lab8ClassifierTrainFlow(FlowSpec):
    """ 
    Train the classifier for iris dataset. 
    """

    @step
    def start(self):
        # Imports 
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        # Load the iris dataset 
        X, y = datasets.load_iris(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X,y, test_size=0.2, random_state=0)
        print("Data loaded successfully, yay!")
        self.next(self.train_rf_classifier, self.train_adaboost_classifier)

    @step
    def train_rf_classifier(self):
        # Train random forest classifier
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def train_adaboost_classifier(self):
        # Train AdaBoost classifier because 
        from sklearn.ensemble import AdaBoostClassifier

        self.model = AdaBoostClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        # Choose a model 
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment('lab8')

        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]

        # Save test data from one of the inputs
        self.test_data = inputs[0].test_data
        self.test_labels = inputs[0].test_labels

        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, artifact_path='metaflow_train', registered_model_name="metaflow-iris-model")

        self.next(self.end)

    @step
    def end(self):
        # Print scores
        print('Scores:')
        print('\n'.join('%s %f' % (type(res[0]).__name__, res[1]) for res in self.results))
        print('Best model:', type(self.model).__name__)

        print('Test data shape:', len(self.test_data), 'labels:', len(self.test_labels))


if __name__=='__main__':
    Lab8ClassifierTrainFlow()