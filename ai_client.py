"""
Thin wrapper around the Anthropic-compatible API.
Handles: plain chat, and vision (photo -> extracted text + explanation).
"""
import base64
import json
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL, BOT_NAME

client = Anthropic(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL)

BASE_SYSTEM_PROMPT = f"""អ្នកគឺជា {{ai_name}} — មិនមែនគ្រាន់តែជា AI ជួយសិក្សាទេ តែជា "មិត្តភ័ក្តិ AI" ដ៏ស្និទ្ធស្នាលរបស់ {{student_name}} សិស្សថ្នាក់ទី {{grade}} ជំនាញ {{track}} នៅកម្ពុជា។

សុទិដ្ឋិនិយម និងបុគ្គលិកលក្ខណៈ៖
- និយាយភាសាខ្មែរជាចម្បង លុះត្រាតែសិស្សសរសេរជាភាសាអង់គ្លេស។
- និយាយដូចមិត្តភ័ក្តិសម័យទំនើប៖ កក្រើក រីករាយ ជូនកំសាន្តតិចៗ (emoji ត្រឹមត្រូវ មិនច្រើនពេក) ប៉ុន្តែស្មោះត្រង់ និងអាចទុកចិត្តបាន។
- ចាំព័ត៌មាន/ចំណូលចិត្ត/អារម្មណ៍ដែលសិស្សធ្លាប់ប្រាប់ពីមុន (មើលផ្នែក "អ្វីដែលចាំបាន" ខាងក្រោម) ហើយប្រើប្រាស់វាដោយធម្មជាតិ ដូចមិត្តភ័ក្តិពិតៗចាំគ្នា — កុំច្រើនពេកដល់ធ្វើឲ្យខ្លាច។
- យល់ចិត្ត (empathize): សង្កេតមើលអារម្មណ៍តាមរបៀបនិយាយ បើសិស្សស្ត្រេស ខឹង ឬអស់សង្ឃឹម សូមឆ្លើយតបដោយកក់ក្តៅ ស្តាប់ជាមុនសិន មុននឹងផ្តល់ដំបូន្មាន។ បើសិស្សរីករាយ សូមរីករាយជាមួយ។
- បើសិស្សសួរអ្វីក្រៅមុខវិជ្ជា (ជីវិត សេចក្តីស្រលាញ់ ហ្គេម ព័ត៌មាន...) អ្នកអាចជជែកធម្មតាបានដូចមិត្តភ័ក្តិ។

ការសិក្សា (នៅតែជាកិច្ចការចម្បង)៖
- ពន្យល់មេរៀន គណិតវិទ្យា រូបវិទ្យា គីមីវិទ្យា ជីវវិទ្យា ភូមិវិទ្យា ប្រវត្តិវិទ្យា និងមុខវិជ្ជាផ្សេងទៀត ក្នុងកម្មវិធីសិក្សាកម្ពុជា ដោយពន្យល់ជាជំហានៗ (step by step) ច្បាស់លាស់ងាយយល់។
- សម្រាប់លំហាត់គណិតវិទ្យា បង្ហាញរូបមន្ត ការគណនា និងចម្លើយចុងក្រោយឲ្យច្បាស់។
- ជួយកាត់បន្ថយស្ត្រេសរឿងសិក្សា ប៉ុន្តែនៅតែជាអ្នកជំនួយការសិក្សាដ៏ជឿទុកចិត្តបាន។
{{memory_block}}"""

def _memory_block(facts):
    if not facts:
        return ""
    bullet_list = "\n".join(f"- {f}" for f in facts)
    return f"\nអ្វីដែលចាំបានអំពី {{student_name}} (ប្រើដោយធម្មជាតិ កុំច្រើនពេក)៖\n{bullet_list}\n"

def _resolve_system(user, facts=None):
    ai_name = (user or {}).get("ai_name") or BOT_NAME
    grade = (user or {}).get("grade") or "12"
    track = (user or {}).get("track") or ""
    student_name = (user or {}).get("name") or "សិស្ស"
    memory_block = _memory_block(facts or [])
    prompt = BASE_SYSTEM_PROMPT.format(
        ai_name=ai_name, grade=grade, track=track,
        student_name=student_name, memory_block=memory_block,
    )
    return prompt.replace("{student_name}", student_name)

def chat(user, history, new_user_message, facts=None):
    """
    history: list of {"role": "user"/"assistant", "content": str}
    facts: list of remembered fact strings (long-term memory) to weave into the system prompt
    """
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": new_user_message})

    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1500,
        system=_resolve_system(user, facts),
        messages=messages,
    )
    return "".join(block.text for block in resp.content if block.type == "text")

def read_image_and_answer(user, image_bytes, media_type, instruction, facts=None):
    """
    Sends a photo (math/khmer test page, etc.) plus an instruction
    (e.g. 'extract the text', 'turn this into a test', 'explain the answer').
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2000,
        system=_resolve_system(user, facts),
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


# --- Long-term memory: pull durable facts out of a message exchange ---

FACT_EXTRACT_SYSTEM = """អ្នកជាឧបករណ៍ស្រង់ព័ត៌មានស្ងាត់ៗ (memory extractor) មិនមែនជាមិត្តជជែកទេ។
ពីការសន្ទនារវាងសិស្ស និង AI ខាងក្រោម សូមស្រង់ចេញនូវ "ការពិត/ចំណូលចិត្ត/ស្ថានភាពរយៈពេលវែង" ចំនួន 0-2
ដែលសមនឹងចងចាំសម្រាប់ការជជែកលើកក្រោយ (ឧ. ចូលចិត្តអ្វី, គោលដៅ, បញ្ហាដែលកំពុងតស៊ូ, ព័ត៌មានផ្ទាល់ខ្លួនស្ថេរភាព)។
កុំរួមបញ្ចូលរឿងបណ្តោះអាសន្ន (សំណួរតែម្តង, អារម្មណ៍ថ្ងៃនេះតែម្នាក់ឯង)។
ឆ្លើយតបជា JSON array នៃ string ខ្លីៗប៉ុណ្ណោះ ដូចជា ["ចូលចិត្តគណិតវិទ្យា ជាពិសេសធរណីមាត្រ", "កំពុងត្រៀមប្រឡងចូលសាកលវិទ្យាល័យខែក្រោយ"]។
បើគ្មានអ្វីសមនឹងចាំទេ សូមឆ្លើយ []។ កុំសរសេរអត្ថបទផ្សេងក្រៅពី JSON array នេះ។"""

def extract_facts(user_message, ai_reply):
    """
    Cheap best-effort pass to spot durable facts worth remembering long-term.
    Returns a list of 0-2 short fact strings (possibly empty). Never raises —
    memory extraction failing should never break the actual chat reply.
    """
    try:
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=200,
            system=FACT_EXTRACT_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"សិស្ស: {user_message}\nAI: {ai_reply}",
            }],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").removeprefix("json").strip()
        facts = json.loads(raw)
        if isinstance(facts, list):
            return [str(f).strip() for f in facts if str(f).strip()][:2]
    except Exception:
        pass
    return []


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
