# Migrating from Heretic 1.x to Heretic 2.x

Heretic 2.x is not backwards compatible with Heretic 1.x configurations.

If you have existing configuration files written to work with a Heretic 1.x release,
you need to make the following changes before using them with a Heretic 2.x release.
Only settings that are actually in your configuration file must be updated;
defaults are automatically set to the current values. Of course, you can also simply
start with [an up-to-date default configuration](/config.default.toml), and modify
it to match your requirements.


## Remove settings for research features

Research features have been removed in Heretic 2.0. The following settings
are no longer recognized:

* `print_residual_geometry`
* `plot_residuals`
* `residual_plot_path`
* `residual_plot_title`
* `residual_plot_style`
* `residual_plot_label` and `residual_plot_color` in prompt sections


## Remove `kl_divergence_scale` and `kl_divergence_target`

KL divergence balancing has been removed and these settings
are no longer recognized.


## Rename `chain_of_thought_skips` to `chain_of_thought_blocks` and update its structure

The new setting `chain_of_thought_blocks` contains pairs of the form
`[cot_initializer, cot_finalizer]` instead of `[cot_initializer, closed_cot_block]`.


## Move abliteration settings to the `[modifier.Abliteration]` section

This includes the following settings:

* `orthogonalize_direction`
* `row_normalization`
* `full_normalization_lora_rank`
* `winsorization_quantile`
* `[good_prompts]` (renamed to `[modifier.Abliteration.good_prompts]`)
* `[bad_prompts]` (renamed to `[modifier.Abliteration.bad_prompts]`)

> [!IMPORTANT]
>
> Heretic 2.x defaults to using ARA for modifying models, not abliteration.
> To continue using abliteration, edit the `modifiers` setting accordingly.
> To configure ARA, edit the `[modifier.ARA]` section.


## Move refusal evaluation settings to the `[scorer.KeywordRate]` section

This includes the following settings:

* `print_responses`
* `refusal_markers` (renamed to `keyword_markers`)
* `[bad_evaluation_prompts]` (renamed to `[scorer.KeywordRate.prompts]`)


## Rename `[good_evaluation_prompts]` to `[scorer.KLDivergence.prompts]`

KL divergence is now computed by a scorer plugin.
