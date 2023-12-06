from django.shortcuts import HttpResponse
# Create your views here.

def generate(request):
    return HttpResponse("This is the Generator Page")