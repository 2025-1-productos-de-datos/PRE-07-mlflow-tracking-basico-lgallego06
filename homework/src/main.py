import uuid
import mlflow
import mlflow.sklearn   

from homework.src._internals.calculate_metrics import calculate_metrics
from homework.src._internals.parse_argument import parse_argument
from homework.src._internals.prepare_data import prepare_data
from homework.src._internals.print_metrics import print_metrics
from homework.src._internals.save_model_if_better import save_model_if_better
from homework.src._internals.select_model import select_model

FILE_PATH = "data/winequality-red.csv"
TEST_SIZE = 0.25
RANDOM_STATE = 123456


def main():

    args = parse_argument()
    model = select_model(args)

    x_train, x_test, y_train, y_test = prepare_data(
        file_path=FILE_PATH,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    
    #se inicia el experimento en MLflow
    mlflow.set_experiment("wine_quality_experiment")
    run_name=f"{args.model}_+{uuid.uuid4().hex[:8]}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("file_path", FILE_PATH)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("model", args.model)
        
        if args.model == "elasticnet":
            mlflow.log_param("alpha", args.alpha)
            mlflow.log_param("l1_ratio", args.l1_ratio)
        elif args.model == "knn":
            mlflow.log_param("n_neighbors", args.n_neighbors)
    
    model.fit(x_train, y_train)

    mse, mae, r2 = calculate_metrics(model, x_train, y_train)
    print_metrics("Training metrics", mse, mae, r2)
    
    #log meticas de entrenamiento
    mlflow.log_metric("mse_train", mse)
    mlflow.log_metric("mae_train", mae) 
    mlflow.log_metric("r2_train", r2)

    mse, mae, r2 = calculate_metrics(model, x_test, y_test)
    print_metrics("Testing metrics", mse, mae, r2)
    
    #log meticas de test
    mlflow.log_metric("mse_test", mse)
    mlflow.log_metric("mae_test", mae)
    mlflow.log_metric("r2_test", r2)
    
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=x_train[:1],
    )


if __name__ == "__main__":
    main()
