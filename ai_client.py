"""
Thin wrapper around the Anthropic-compatible API.
Handles: plain chat, and vision (photo -> extracted text + explanation).
"""
import base64
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL, BOT_NAME

client = Anthropic(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL)

BASE_SYSTEM_PROMPT = f"""អ្នកគឺជា {{ai_name}} ជំនួយការសិក្សា AI សម្រាប់សិស្សខ្មែរនៅកម្ពុជា ថ្នាក់ទី {{grade}} ជំនាញ {{track}}។
- និយាយភាសាខ្មែរជាចម្បង លុះត្រាតែសិស្សសរសេរជាភាសាអង់គ្លេស។
- ពន្យល់មេរៀន គណិតវិទ្យា រូបវិទ្យា គីមីវិទ្យា ជីវវិទ្យា ភូមិវិទ្យា ប្រវត្តិវិទ្យា និងមុខវិជ្ជាផ្សេងទៀត ក្នុងកម្មវិធីសិក្សាកម្ពុជា ដោយពន្យល់ជាជំហានៗ (step by step) ច្បាស់លាស់ងាយយល់។
- សម្រាប់លំហាត់គណិតវិទ្យា បង្ហាញរូបមន្ត ការគណនា និងចម្លើយចុងក្រោយឲ្យច្បាស់។
- និយាយស្និទ្ធស្នាល កម្សាន្ត ជួយកាត់បន្ថយស្ត្រេស ប៉ុន្តែនៅតែជាអ្នកជំនួយការសិក្សាដ៏ជឿទុកចិត្តបាន។
- បើសិស្សសួរអ្វីក្រៅមុខវិជ្ជា អ្នកអាចជួយបានដែរ ដូចជា Claude ធម្មតា។
"""

def _resolve_system(user):
    ai_name = (user or {}).get("ai_name") or BOT_NAME
    grade = (user or {}).get("grade") or "12"
    track = (user or {}).get("track") or ""
    return BASE_SYSTEM_PROMPT.format(ai_name=ai_name, grade=grade, track=track)

def chat(user, history, new_user_message):
    """
    history: list of {"role": "user"/"assistant", "content": str}
    """
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": new_user_message})

    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1500,
        system=_resolve_system(user),
        messages=messages,
    )
    return "".join(block.text for block in resp.content if block.type == "text")

def read_image_and_answer(user, image_bytes, media_type, instruction):
    """
    Sends a photo (math/khmer test page, etc.) plus an instruction
    (e.g. 'extract the text', 'turn this into a test', 'explain the answer').
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2000,
        system=_resolve_system(user),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


# --- Reusable instructions for the button menu ---

INSTR_EXTRACT = (
    "សូមអានអត្ថបទ/សំណួរទាំងអស់ក្នុងរូបភាពនេះឲ្យបានត្រឹមត្រូវ (OCR) ហើយសរសេរជាអក្សរខ្មែរ/គណិតវិទ្យាឡើងវិញ "
    "ដោយមិនកែប្រែខ្លឹមសារ។ បើជាលំហាត់គណិតវិទ្យា សរសេររូបមន្តឲ្យច្បាស់។"
)

INSTR_MAKE_TEST = (
    "ផ្អែកលើខ្លឹមសារក្នុងរូបភាព/អត្ថបទនេះ សូមបង្កើតជាកម្រងសំណួរតេស្ត (quiz) ចំនួន 5-10 សំណួរ "
    "ដែលមានទាំងសំណួរជម្រើសពហុភាព (multiple choice) និងសំណួរចម្លើយខ្លី សម្រាប់សិស្សត្រួតពិនិត្យចំណេះដឹងខ្លួនឯង។ "
    "កុំបង្ហាញចម្លើយភ្លាមៗ ដាក់ចម្លើយនៅផ្នែកចុងក្រោយ ដាក់ក្បាល 'ចម្លើយ'."
)

INSTR_MAKE_QUESTION = (
    "ផ្អែកលើខ្លឹមសារនេះ សូមបង្កើតសំណួរអនុវត្តន៍ថ្មីៗ (practice questions) ដែលស្រដៀងគ្នាតែមិនដូចគ្នាបេះបិទ "
    "ដើម្បីឲ្យសិស្សអនុវត្តន៍បន្ថែម។"
)

INSTR_SUMMARIZE = (
    "សូមសង្ខេបខ្លឹមសារសំខាន់ៗក្នុងរូបភាព/អត្ថបទនេះ ជាចំណុចៗ (bullet points) ខ្លីៗងាយចាំ សម្រាប់ត្រៀមប្រឡង។"
)

INSTR_EXPLAIN_ANSWER = (
    "រូបភាពនេះជាលំហាត់គណិតវិទ្យា ឬសំណួរដែលត្រូវការចម្លើយ។ សូម៖\n"
    "1) សរសេរឡើងវិញនូវសំណួរ/ប្រធានបទ\n"
    "2) ដោះស្រាយជាជំហានៗ ច្បាស់លាស់ (step-by-step)\n"
    "3) ផ្តល់ចម្លើយចុងក្រោយឲ្យច្បាស់\n"
    "4) ពន្យល់ថាហេតុអ្វីបានចម្លើយនោះត្រូវ (គំនិត/ច្បាប់ដែលប្រើ)"
)
