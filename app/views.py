import json
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .ai_logic import get_response, initialize_llm

@csrf_exempt
def AIView(request):
    if request.method != 'POST':
        return JsonResponse({'message': 'Method Not Allowed'}, status=405)

    question = request.POST.get('question')
    thread_id = request.POST.get('thread_id')

    
    def event_stream():
        initialize_llm()
        for content in get_response(question, thread_id=thread_id):
            if content:
                yield f"data: {json.dumps({'text': content})}\n\n"
        yield "event: complete\ndata: {}\n\n"
                    
    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')
