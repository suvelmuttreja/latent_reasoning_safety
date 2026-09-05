# Third-party sources and notices

The [MIT license](LICENSE) applies to this project's original code and writing.
It does not replace the licenses of upstream code, benchmark questions, model
weights, or other third-party material. Generated responses are retained as
experimental observations, with their model and prompt provenance.

| Material | Source used by this project | Notice |
| --- | --- | --- |
| Coconut algorithm and wrapper reference | [facebookresearch/coconut at 27273cb](https://github.com/facebookresearch/coconut/tree/27273cb8cca4bb763c041a63b036d0c3b7cbbb48) | [Meta MIT notice](licenses/coconut-MIT.txt), retained for the implementation derived from this reference. |
| StrongREJECT evaluator | [dsbowen/strong_reject at 7a551d5](https://github.com/dsbowen/strong_reject/tree/7a551d5b440ec7b75d4f6f5bb7c1719965b76b47) | [Dillon Bowen MIT notice](licenses/strong-reject-MIT.txt). Installed as a pinned dependency. |
| StrongREJECT-small prompts | [alexandrasouly/strongreject at f7cad6c](https://github.com/alexandrasouly/strongreject/tree/f7cad6c17e624e21d8df2278e918ae1dddb4cb56) | [Center for Human-Compatible AI MIT notice](licenses/strongreject-data.txt). Questions appear in manifests and generation records. |
| GSM8K questions and answers | [OpenAI GSM8K at 740312a](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866), originating in [grade-school-math](https://github.com/openai/grade-school-math) | [OpenAI MIT notice](licenses/gsm8k-MIT.txt). Questions/answers appear in manifests and generation records. |

The trained models start from
[Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507);
small validation runs use Qwen3-0.6B. The evaluator uses the model identified in
[configs/evaluation.yaml](configs/evaluation.yaml). Model weights are not
redistributed here; consult the respective model cards and licenses before
using or redistributing them.

Additional engineering and research references, including `wassname/coconut`
and `BatsResearch/self-jailbreaking`, are pinned in
[configs/pins.json](configs/pins.json). The setup helper places their original
repositories, with their notices, in the ignored `vendor/` directory.

The papers motivating the experiment are cited in the
[write-up](writeup/FULL_WRITEUP.built.md). Repository citation metadata is in
[CITATION.cff](CITATION.cff).
