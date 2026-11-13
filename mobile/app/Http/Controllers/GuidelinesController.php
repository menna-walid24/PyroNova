<?php

namespace App\Http\Controllers;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\DB;
use Illuminate\Http\Request;
class GuidelinesController extends Controller
{
 
    //show all guidlines
    public function show(){
return DB::table('guidlines')->orderBy('step_order')->get();
    }



//select means changing it's state
    public function select(Request $request){
        $request->validate([
            "guidline_id"=>'required|exists:guidlines,id'
        ]);

$user=$request->user();
$gid_id=$request->guidline_id;
DB::table('user_guidlines')->insertOrIgnore(['guidline_id'=>$gid_id,'user_id'=>$user->id])
return respone()->json([
    "message"=>'Marked Successfully';

]);


    }
    //show only marked by users 
    public function mark(Request $request){
        $user=$request->user();
        DB::table('guidlines')
    ->join('user_guidlines', 'guidlines.id', '=', 'user_guidlines.guidline_id')->where('user_guidlines.user_id', $user->id)
        ->orderBy('guidlines.step_order')
        ->select('guidlines.*')
        ->get();
    }
}



