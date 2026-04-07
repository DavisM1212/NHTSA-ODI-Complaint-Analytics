from src.modeling.component_multilabel import (
    build_catboost_model,
    is_retryable_catboost_gpu_error,
)


def test_is_retryable_catboost_gpu_error_detects_known_cuda_signature():
    message = 'catboost/cuda/targets/gpu_metrics.cpp:513: Condition violated: `approx[dim].size() == target.size()`'
    assert is_retryable_catboost_gpu_error(RuntimeError(message)) is True


def test_is_retryable_catboost_gpu_error_ignores_generic_messages():
    assert is_retryable_catboost_gpu_error(RuntimeError('ordinary failure')) is False


def test_build_catboost_model_multilabel_does_not_request_hammingloss_custom_metric():
    model = build_catboost_model(
        task_type='GPU',
        devices='0',
        random_seed=42,
        verbose=0,
        iterations=20
    )
    params = model.get_params()

    assert params.get('loss_function') == 'MultiLogloss'
    assert params.get('eval_metric') == 'MultiLogloss'
    assert 'custom_metric' not in params
