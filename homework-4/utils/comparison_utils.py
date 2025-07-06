import os

import numpy as np
import pandas as pd


def compare_models_performance(results_dict, save_path=None):
    """Создает сводную таблицу сравнения моделей"""
    comparison_data = []

    for model_name, results in results_dict.items():
        comparison_data.append(
            {
                "Model": model_name,
                "Best Test Accuracy": f"{max(results['test_accs']):.4f}",
                "Final Test Accuracy": f"{results['test_accs'][-1]:.4f}",
                "Best Train Accuracy": f"{max(results['train_accs']):.4f}",
                "Final Train Accuracy": f"{results['train_accs'][-1]:.4f}",
                "Overfitting Gap": f"{max(results['train_accs']) - max(results['test_accs']):.4f}",
                "Training Time (s)": f"{results.get('total_time', 0):.2f}",
                "Parameters (M)": f"{results.get('parameters', {}).get('total_params_millions', 0):.2f}",
                "Inference Time (ms)": f"{results.get('inference_time', {}).get('mean_time', 0) * 1000:.2f}",
            }
        )

    df = pd.DataFrame(comparison_data)

    if save_path:
        df.to_csv(save_path, index=False)

    return df


def analyze_overfitting(results_dict):
    """Анализирует переобучение моделей"""
    overfitting_analysis = {}

    for model_name, results in results_dict.items():
        train_acc = max(results["train_accs"])
        test_acc = max(results["test_accs"])
        overfitting_gap = train_acc - test_acc

        overfitting_analysis[model_name] = {
            "max_train_acc": train_acc,
            "max_test_acc": test_acc,
            "overfitting_gap": overfitting_gap,
            "overfitting_ratio": overfitting_gap / train_acc if train_acc > 0 else 0,
        }

    return overfitting_analysis


def calculate_efficiency_metrics(results_dict):
    """Рассчитывает метрики эффективности моделей"""
    efficiency_metrics = {}

    for model_name, results in results_dict.items():
        params_millions = results.get("parameters", {}).get("total_params_millions", 0)
        inference_time = results.get("inference_time", {}).get("mean_time", 0)
        best_acc = max(results["test_accs"])

        efficiency_metrics[model_name] = {
            "accuracy_per_param": (
                best_acc / params_millions if params_millions > 0 else 0
            ),
            "accuracy_per_time": best_acc / inference_time if inference_time > 0 else 0,
            "params_per_accuracy": (
                params_millions / best_acc if best_acc > 0 else float("inf")
            ),
            "time_per_accuracy": (
                inference_time / best_acc if best_acc > 0 else float("inf")
            ),
        }

    return efficiency_metrics


def statistical_significance_test(results_dict, metric="test_accs"):
    """Проводит статистический тест значимости различий между моделями"""
    from scipy import stats

    model_names = list(results_dict.keys())
    n_models = len(model_names)

    # Создаем матрицу результатов для ANOVA
    all_results = []
    group_labels = []

    for model_name in model_names:
        results = results_dict[model_name][metric]
        all_results.extend(results)
        group_labels.extend([model_name] * len(results))

    # ANOVA тест
    groups = [results_dict[name][metric] for name in model_names]
    f_stat, p_value = stats.f_oneway(*groups)

    # Попарные t-тесты
    pairwise_tests = {}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1, model2 = model_names[i], model_names[j]
            t_stat, p_val = stats.ttest_ind(
                results_dict[model1][metric], results_dict[model2][metric]
            )
            pairwise_tests[f"{model1}_vs_{model2}"] = {
                "t_statistic": t_stat,
                "p_value": p_val,
                "significant": p_val < 0.05,  # type: ignore
            }

    return {
        "anova_f_stat": f_stat,
        "anova_p_value": p_value,
        "pairwise_tests": pairwise_tests,
    }


def create_model_ranking(
    results_dict, metrics=["test_accs", "total_time", "total_params_millions"]
):
    """Создает рейтинг моделей по различным метрикам"""
    ranking_data = {}

    for metric in metrics:
        if metric == "test_accs":
            # Для точности - чем больше, тем лучше
            sorted_models = sorted(
                results_dict.items(), key=lambda x: max(x[1][metric]), reverse=True
            )
        else:
            # Для времени и параметров - чем меньше, тем лучше
            sorted_models = sorted(
                results_dict.items(),
                key=lambda x: (
                    x[1].get(metric, float("inf"))
                    if isinstance(x[1].get(metric, float("inf")), (int, float))
                    else float("inf")
                ),
            )

        ranking_data[metric] = {
            model_name: rank + 1 for rank, (model_name, _) in enumerate(sorted_models)
        }

    # Общий рейтинг (среднее по всем метрикам)
    overall_ranking = {}
    for model_name in results_dict.keys():
        ranks = []
        for metric in metrics:
            if metric in ranking_data:
                ranks.append(ranking_data[metric].get(model_name, len(results_dict)))
        overall_ranking[model_name] = np.mean(ranks)

    # Сортируем по общему рейтингу
    overall_ranking = dict(sorted(overall_ranking.items(), key=lambda x: x[1]))

    return {"metric_rankings": ranking_data, "overall_ranking": overall_ranking}


def analyze_learning_curves(results_dict):
    """Анализирует кривые обучения моделей"""
    learning_analysis = {}

    for model_name, results in results_dict.items():
        train_accs = results["train_accs"]
        test_accs = results["test_accs"]

        # Скорость обучения (наклон в начале)
        early_train_slope = (
            (train_accs[4] - train_accs[0]) / 4 if len(train_accs) > 4 else 0
        )
        early_test_slope = (
            (test_accs[4] - test_accs[0]) / 4 if len(test_accs) > 4 else 0
        )

        # Стабильность (дисперсия в конце)
        late_train_stability = (
            np.var(train_accs[-5:]) if len(train_accs) >= 5 else np.var(train_accs)
        )
        late_test_stability = (
            np.var(test_accs[-5:]) if len(test_accs) >= 5 else np.var(test_accs)
        )

        # Сходимость (разница между последними эпохами)
        convergence_train = (
            abs(train_accs[-1] - train_accs[-2]) if len(train_accs) > 1 else 0
        )
        convergence_test = (
            abs(test_accs[-1] - test_accs[-2]) if len(test_accs) > 1 else 0
        )

        learning_analysis[model_name] = {
            "early_train_slope": early_train_slope,
            "early_test_slope": early_test_slope,
            "late_train_stability": late_train_stability,
            "late_test_stability": late_test_stability,
            "convergence_train": convergence_train,
            "convergence_test": convergence_test,
            "learning_efficiency": (
                early_test_slope / early_train_slope if early_train_slope != 0 else 0
            ),
        }

    return learning_analysis


def generate_comprehensive_report(results_dict, save_dir="results"):
    """Генерирует комплексный отчет по всем экспериментам"""
    os.makedirs(save_dir, exist_ok=True)

    # Создаем все анализы
    comparison_df = compare_models_performance(
        results_dict, os.path.join(save_dir, "model_comparison.csv")
    )
    overfitting_analysis = analyze_overfitting(results_dict)
    efficiency_metrics = calculate_efficiency_metrics(results_dict)
    ranking = create_model_ranking(results_dict)
    learning_analysis = analyze_learning_curves(results_dict)

    # Сохраняем результаты анализа
    with open(os.path.join(save_dir, "analysis_report.txt"), "w") as f:
        f.write("=== COMPREHENSIVE MODEL ANALYSIS REPORT ===\n\n")

        f.write("1. MODEL COMPARISON TABLE:\n")
        f.write(comparison_df.to_string())
        f.write("\n\n")

        f.write("2. OVERFITTING ANALYSIS:\n")
        for model_name, analysis in overfitting_analysis.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Overfitting Gap: {analysis['overfitting_gap']:.4f}\n")
            f.write(f"  Overfitting Ratio: {analysis['overfitting_ratio']:.4f}\n")
        f.write("\n")

        f.write("3. EFFICIENCY METRICS:\n")
        for model_name, metrics in efficiency_metrics.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Accuracy per Parameter: {metrics['accuracy_per_param']:.6f}\n")
            f.write(f"  Accuracy per Time: {metrics['accuracy_per_time']:.6f}\n")
        f.write("\n")

        f.write("4. MODEL RANKING:\n")
        f.write("Overall Ranking (lower is better):\n")
        for model_name, rank in ranking["overall_ranking"].items():
            f.write(f"  {model_name}: {rank:.2f}\n")
        f.write("\n")

        f.write("5. LEARNING CURVE ANALYSIS:\n")
        for model_name, analysis in learning_analysis.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Learning Efficiency: {analysis['learning_efficiency']:.4f}\n")
            f.write(f"  Test Stability: {analysis['late_test_stability']:.6f}\n")
        f.write("\n")

    return {
        "comparison_df": comparison_df,
        "overfitting_analysis": overfitting_analysis,
        "efficiency_metrics": efficiency_metrics,
        "ranking": ranking,
        "learning_analysis": learning_analysis,
    }
