import json
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .ai_logic import get_response , get_all_projects

@csrf_exempt
def AIView(request):
    if request.method != 'POST':
        return JsonResponse({'message': 'Method Not Allowed'}, status=405)

    question = request.POST.get('question')
    thread_id = request.POST.get('thread_id')

    
    def event_stream():
        for content in get_response(question, thread_id=thread_id):
            if content:
                yield f"data: {json.dumps({'text': content})}\n\n"
        yield "event: complete\ndata: {}\n\n"
                    
    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')

@csrf_exempt
def ProjectListView(request):
    print("Fetching data from ChromaDB...")
    try:
        data = get_all_projects()
        return JsonResponse(data , safe=False)
    except Exception as e:
        return JsonResponse({"error":str(e)},status=500)