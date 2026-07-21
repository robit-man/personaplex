import torch

from ground_truth_finetuning.tools.train_arc4_reference import crop_native_turn


def test_crop_retains_recent_context_and_every_target() -> None:
    codes = torch.arange(2 * 1000).reshape(2, 1000)
    mask = torch.zeros_like(codes, dtype=torch.bool)
    mask[:, 700:801] = True
    cropped_codes, cropped_mask, prefix_at = crop_native_turn(
        codes,
        mask,
        700,
        context_frames=256,
        post_target_tail_frames=16,
    )
    assert cropped_codes.shape == (2, 373)
    assert prefix_at == 256
    assert int(cropped_mask.sum()) == int(mask.sum())
    torch.testing.assert_close(cropped_codes[:, 0], codes[:, 444])
    torch.testing.assert_close(cropped_codes[:, -1], codes[:, 816])


def test_crop_rejects_target_before_control_boundary() -> None:
    codes = torch.zeros(2, 100)
    mask = torch.zeros_like(codes, dtype=torch.bool)
    mask[:, 40:50] = True
    try:
        crop_native_turn(
            codes,
            mask,
            45,
            context_frames=20,
            post_target_tail_frames=2,
        )
    except ValueError as error:
        assert "before the control boundary" in str(error)
    else:
        raise AssertionError("crop accepted pre-control target supervision")
